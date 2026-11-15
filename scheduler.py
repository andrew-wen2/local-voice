"""Turn scheduler: which piece of GPU work runs next (ARCHITECTURE.md §5.9). No MLX here: the GPU work comes
from a backend, as generators of pieces, so the same logic runs on the GPU or on a fake clock in tests.

The context for a question is four regions, oldest first:
  prefix     <bos>, the timeline of the hour (its own turn), the conversation so far, the pages turn's opening:
             an exact cache. Its version goes up when it is rebuilt (a visit ended: quiet time only, so the
             timeline is fixed for a turn) or when a finished exchange is appended (the next answer waits for that)
  pages      retrieved pages in context order: the first `frontier` computed exactly (together, in order, after
             the timeline), the rest from stored graph pieces (§3.11)
  screen     the newest finished screen read (a stored piece)
  question   prefilled ahead of key up when its inputs are stable
A key names one context: (prefix version, pages, frontier, screen frame). Work done against a key that has
since changed is stale and is dropped; nothing stale reaches an answer.

States: QUIET -key down-> LISTENING -mic silent-> TAIL -key up-> RELEASED -first token-> ANSWERING -done-> QUIET,
and TAIL -speech-> LISTENING.

Classes, highest first; the loop runs one piece of the highest-class runnable job (ties: order, then age):
  CRITICAL    the answer: assemble, question tail, decode                       RELEASED, ANSWERING
  BLOCKING    a screen read (under rule B the next answer waits for it); appending a finished exchange to the
              prefix; the first prefix build                                    every state
  IMPROVING   stored piece for a retrieved page that has none (order 0); exact upgrade of the pages, LISTENING
              only (1); question prefill, in TAIL or once the upgrade is done (2)          LISTENING, TAIL
  BACKGROUND  graph builds, prefix rebuild                                      QUIET
Piece limits: an IMPROVING or BACKGROUND piece starts only if its estimated time fits the state's limit, so the
piece in flight at key up is short. CRITICAL and BLOCKING pieces have no limit: the answer needs that work anyway.
After key up, until the final transcript is in (~40 ms, the answer can't start before it), IMPROVING work for the
current context still runs under TAIL's limit; a question prefill already under way that the final transcript keeps
then finishes as CRITICAL before the answer (M3, 2026-11-15).

Backend protocol. Each work method returns a generator that yields the kind of its *next* piece before doing it
and does no work before its first yield:
    def read(fid):  yield "prep"; <prep>; yield "vision"; <vision layer>; ...
The scheduler primes it (first next(): no work), checks the announced kind against the limit, then each next()
runs exactly one piece. Methods: read(fid), build(page), prefix(n_exchanges), append(i), upgrade(pages, i),
spec(key, ids, have), answer(plan) (returns {"exchange": bool, "rebuild": bool}); has_piece(page); prefetch(pages);
question_ids(text); `priors` (seconds per piece kind) and optionally `prior(kind)` for kinds not in priors.
"""
from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Callable, Optional


class State(Enum):
    QUIET = "quiet"
    LISTENING = "listening"
    TAIL = "tail"
    RELEASED = "released"
    ANSWERING = "answering"


class Cls(IntEnum):
    CRITICAL = 0
    BLOCKING = 1
    IMPROVING = 2
    BACKGROUND = 3


ALLOWED = {
    State.QUIET: {Cls.BLOCKING, Cls.BACKGROUND},
    State.LISTENING: {Cls.BLOCKING, Cls.IMPROVING},
    State.TAIL: {Cls.BLOCKING, Cls.IMPROVING},
    State.RELEASED: {Cls.CRITICAL, Cls.BLOCKING},
    State.ANSWERING: {Cls.CRITICAL, Cls.BLOCKING},
}

# Longest IMPROVING / BACKGROUND piece allowed to start, seconds. Provisional (§5.9): to be set from the measured
# gap between the last word and key up, before any latency run.
PIECE_LIMIT_S = {State.QUIET: math.inf, State.LISTENING: 0.050, State.TAIL: 0.025,
                 State.RELEASED: 0.0, State.ANSWERING: 0.0}


class Estimates:
    """Running estimate of each piece kind's time: seeded from priors (or `fallback(kind)`, e.g. a cost per token
    for kinds named by size), then the lower median of the prior and the last SAMPLES times. A kind with neither
    raises: a piece that is never allowed to run could otherwise never be measured.

    M3 (2026-11-15): it was an average weighted 0.3 to the newest. One slow piece (a page-in under memory pressure:
    64-300 ms for a ~5 ms retrieval piece) then pushed its kind's estimate over the state's limit; a kind over the
    limit never runs, so it was never measured again, and the job waited for the rest of the turn (every partial's
    retrieval in a whole replay). Now one outlier doesn't move the estimate, and a kind held back by the limit
    forgets its largest time every BLOCKS_TO_FORGET picks it loses, so it is retried and re-measured."""
    SAMPLES, BLOCKS_TO_FORGET = 5, 20

    def __init__(self, priors: dict[str, float], fallback: Optional[Callable[[str], float]] = None):
        self.t, self.fallback = dict(priors), fallback
        self._samples: dict[str, deque] = {}
        self._blocks: dict[str, int] = {}

    def get(self, kind: str) -> float:
        if kind not in self.t and self.fallback is not None:
            self.t[kind] = self.fallback(kind)
        return self.t[kind]

    def _recompute(self, kind: str):
        xs = sorted(self._samples[kind])
        self.t[kind] = xs[(len(xs) - 1) // 2]           # lower median

    def update(self, kind: str, dt: float):
        if kind not in self._samples:
            seed = [self.t[kind]] if kind in self.t else []
            self._samples[kind] = deque(seed, maxlen=self.SAMPLES + 1)
        self._samples[kind].append(dt)
        self._blocks[kind] = 0
        self._recompute(kind)

    def blocked(self, kind: str):
        """The kind lost a pick to the piece limit."""
        self._blocks[kind] = self._blocks.get(kind, 0) + 1
        xs = self._samples.get(kind)
        if self._blocks[kind] >= self.BLOCKS_TO_FORGET and xs is not None and len(xs) > 1:
            xs.remove(max(xs))
            self._blocks[kind] = 0
            self._recompute(kind)


@dataclass
class TurnContext:
    timeline: int                       # prefix version at key down (for the log)
    pages: tuple = ()                   # retrieved pages, context order
    frontier: int = 0                   # pages[:frontier] are computed exactly
    words: tuple = ()                   # question token ids so far
    final: bool = False                 # the final transcript is in
    spec: Optional[tuple] = None        # (key, ids) of the finished question prefill
    alt: Optional[tuple] = None         # the runner-up's slots (another kept screen), computed exactly when idle
    want: Optional[str] = None          # newest partial transcript not yet retrieved (with retrieve_steps)
    stats: dict = field(default_factory=lambda: {"retrievals": 0, "page_changes": 0, "pages_cut": 0})


# ---------------------------------------------------------------- jobs

class Job:
    order = 0

    def __init__(self):
        self.gen = None
        self.next_kind: Optional[str] = None
        self.gpu_s = 0.0
        self.seq = 0
        self.ran = False                    # at least one piece has run

    def prime(self, gen):
        self.gen = gen
        self.next_kind = next(gen)          # no work happens before the first yield

    def cls(self, s: "Scheduler") -> Cls:
        raise NotImplementedError

    def runnable(self, s: "Scheduler") -> bool:
        return True

    def stale(self, s: "Scheduler") -> bool:
        return False

    def finish(self, s: "Scheduler", result):
        pass


class Read(Job):
    def __init__(self, s, fid: int):
        super().__init__()
        self.fid = fid
        self.prime(s.backend.read(fid))

    def cls(self, s):
        return Cls.BLOCKING

    def finish(self, s, result):
        s._read_done(self)


class Build(Job):
    """A page's stored graph piece. Needed by the turn (IMPROVING) when the page is retrieved and past the exact
    frontier; otherwise background work for later questions."""

    def __init__(self, s, page):
        super().__init__()
        self.page = page
        self.prime(s.backend.build(page))

    def cls(self, s):
        c = s.ctx
        return Cls.IMPROVING if c is not None and self.page in c.pages[c.frontier:] else Cls.BACKGROUND


class Prefix(Job):
    """Rebuild the prefix from scratch: the timeline as it is now and the first n finished exchanges. Background
    work, except the first build: nothing can be answered without a prefix."""
    order = 1

    def __init__(self, s):
        super().__init__()
        self.n = s.ex_done
        self.prime(s.backend.prefix(self.n))

    def cls(self, s):
        return Cls.BLOCKING if s.prefix == 0 else Cls.BACKGROUND

    def stale(self, s):
        return s.ex_done != self.n                  # an exchange finished meanwhile: rebuild with it

    def finish(self, s, result):
        s.ex_in = self.n
        s._prefix_changed()


class Append(Job):
    """Finished exchange i appended to the prefix. The next answer waits for it."""
    order = 1

    def __init__(self, s):
        super().__init__()
        self.i, self.v = s.ex_in, s.prefix
        self.prime(s.backend.append(self.i))

    def cls(self, s):
        return Cls.BLOCKING

    def stale(self, s):
        return s.ex_in != self.i or s.prefix != self.v

    def finish(self, s, result):
        s.ex_in = self.i + 1
        s._prefix_changed()


def _label(pages) -> Optional[str]:
    """A short readable name for a context's pages (diagnostics): kept screens by frame id, the memory slot by key."""
    if not pages:
        return None
    out = []
    for p in pages:
        if p and str(p[0]).endswith("screen"):
            out.append(f"s{p[1]}")
        elif len(p) == 1:
            out.append(str(p[0])[:6])
        else:
            out.append(str(p[0])[:24])
    return "|".join(out)


class Upgrade(Job):
    """Page i computed exactly after the timeline and pages[:i] (which are exact already)."""
    order = 1

    def __init__(self, s):
        super().__init__()
        c = s.ctx
        self.i, self.target, self.v = c.frontier, c.pages[:c.frontier + 1], s.prefix
        self.prime(s.backend.upgrade(self.target, self.i))

    def cls(self, s):
        return Cls.IMPROVING

    def runnable(self, s):
        # Frozen in TAIL, where only the question prefill helps near key up -- unless the prefill can't run because
        # a page has neither an exact cache nor a stored piece; then this page is the only useful work (M3, 2026-11-15).
        # The same holds after key up until the final transcript is in (the answer can't start yet).
        return s.state is State.LISTENING or (s._near_up() and bool(s.missing()))

    def stale(self, s):
        c = s.ctx
        # a new prefix invalidates the exact cache under it (can't happen mid-turn under today's rules: kept so a
        # change to them can't finish a page onto the old prefix)
        return c is None or c.frontier != self.i or c.pages[:self.i + 1] != self.target or s.prefix != self.v

    def finish(self, s, result):
        s.ctx.frontier = self.i + 1
        s._diag("frontier_log", self.i + 1)


class Retrieve(Job):
    """Retrieval for the newest partial transcript, in pieces (M3, 2026-11-15). It ran synchronously in partial()
    before: ~90 ms a partial of GPU work outside the pieces, 6-10 partials a question, so page builds idled ~30% of
    the time while you spoke. Partials that arrive while one runs are not retrieved; the newest is, next. Retrieval
    of the final transcript stays synchronous (final()), so what is answered from is unchanged."""
    order = -1                                  # before page work: it decides which pages to build

    def __init__(self, s, text: str):
        super().__init__()
        self.text = text
        self.prime(s.retrieve_steps(text))

    def cls(self, s):
        return Cls.IMPROVING

    def runnable(self, s):
        ready = getattr(s.backend, "retrieval_ready", None)     # its next piece's inputs read off the disk
        return (s.state is State.LISTENING or s._near_up()) and (ready is None or ready())

    def stale(self, s):
        return s.ctx is None or s.ctx.final

    def finish(self, s, result):
        pages, alt = result
        s._apply_retrieval(self.text, pages, alt)


class Spec(Job):
    """The question so far, prefilled over the current context; extends a finished prefix when there is one."""
    order = 2

    def __init__(self, s):
        super().__init__()
        self.key, self.ids = s.key(), s.ctx.words
        self.prime(s.backend.spec(self.key, self.ids, s._question_reusable(self.key, self.ids)))

    def cls(self, s):
        # Started and still valid for the final transcript: the answer would redo it from scratch, so it finishes
        # first (M3, 2026-11-15; before, a prefill cut off at key up was thrown away whole).
        return Cls.CRITICAL if s.ctx is not None and s.ctx.final and self.ran else Cls.IMPROVING

    def runnable(self, s):
        return s._spec_phase() or (s.ctx is not None and s.ctx.final and self.ran)

    def stale(self, s):
        c = s.ctx
        return c is None or s.key() != self.key or c.words[:len(self.ids)] != self.ids

    def finish(self, s, result):
        s.ctx.spec = (self.key, self.ids)


class AltUpgrade(Job):
    """The next slot of the runner-up context (`ctx.alt`: the next-best kept screen with its memory), computed exactly
    once the current context and question prefill are done, so a late retrieval switch to it finds it ready (M3,
    2026-11-13: the question's last words often change which screen is used). Needs a backend with branches."""
    order = 3

    def __init__(self, s):
        super().__init__()
        self.target, self.v = s.ctx.alt, s.prefix
        self.i = s.backend.exact_len(self.target)
        self.prime(s.backend.upgrade(self.target, self.i))

    def cls(self, s):
        return Cls.IMPROVING

    def runnable(self, s):
        c = s.ctx
        return s.state is State.LISTENING and c is not None and c.frontier == len(c.pages)

    def stale(self, s):
        c = s.ctx
        return c is None or c.alt != self.target or s.prefix != self.v


class Answer(Job):
    def __init__(self, need: int):
        super().__init__()
        self.need = need                        # screen frame the answer must see (rule B), -1 for none
        self.next_kind = "answer"
        self.plan: Optional[dict] = None

    def cls(self, s):
        return Cls.CRITICAL

    def runnable(self, s):
        return (s.ctx is not None and s.ctx.final and s.screen >= max(self.need, 0)
                and s.prefix > 0 and s.ex_in == s.ex_done
                and not any(isinstance(j, Spec) and j.ran and not j.stale(s) for j in s.jobs))

    def finish(self, s, result):
        s._answer_done(self, result)


# ---------------------------------------------------------------- the scheduler

class Scheduler:
    def __init__(self, backend, retrieve: Callable[[str], tuple], estimates: Optional[Estimates] = None,
                 clock: Callable[[], float] = time.perf_counter, freshness: str = "B",
                 limits: Optional[dict] = None, retrieve_alt: Optional[Callable[[str], Optional[tuple]]] = None,
                 retrieve_steps: Optional[Callable] = None):
        assert freshness in ("B", "C"), freshness       # A answered from the old screen and got it wrong (§5.4)
        self.backend, self.retrieve, self.clock, self.freshness = backend, retrieve, clock, freshness
        self.retrieve_alt = retrieve_alt if hasattr(backend, "exact_len") else None   # branches needed
        self.retrieve_steps = retrieve_steps   # text -> generator of pieces returning (pages, alt): see Retrieve
        self.est = estimates or Estimates(backend.priors, fallback=getattr(backend, "prior", None))
        self.limits = limits or PIECE_LIMIT_S
        self.state = State.QUIET
        self.ctx: Optional[TurnContext] = None
        self.jobs: list[Job] = []
        self.read: Optional[Read] = None
        self.pending_frame: Optional[int] = None        # a frame waiting behind a read an answer needs
        self.answer: Optional[Answer] = None
        self.screen = -1                                # frame id of the newest finished read
        self.newest_frame = -1
        self.prefix = 0                                 # prefix version; 0 = not built yet
        self.prefix_dirty = True                        # the timeline changed: rebuild in quiet time
        self.ex_done = 0                                # finished exchanges
        self.ex_in = 0                                  # of them, in the prefix
        self.turn: Optional[dict] = None
        self.turns: list[dict] = []
        self.log: deque = deque(maxlen=20000)           # (t0, dt, state, cls, kind, estimate), the latest pieces
        self.on_turn: Optional[Callable[[dict], None]] = None   # called with each closed turn's log
        self.stats = {"reads_dropped": 0, "wasted_s": 0.0, "pieces": 0}
        self.errors: list[tuple] = []                   # (job, piece kind, traceback), newest last
        self._seq = 0

    # ------------------------------------------------------------ events (the driver calls these between pieces)

    def key_down(self, t: Optional[float] = None):
        if self.answer is not None:                     # barge-in: the old answer stops
            self._drop(self.answer)
            self._close_turn(cancelled=True)
        self.state = State.LISTENING
        self.ctx = TurnContext(timeline=self.prefix)
        self.turn = {"t_down": self.clock() if t is None else t, "gpu_s": {c.name: 0.0 for c in Cls},
                     "wasted_s": 0.0, "prefix_at_down": self.prefix,
                     # diagnostics (M3): state changes, GPU time per state:class:kind, pieces the limit held back
                     "states": [], "pieces": {}, "blocked": {}}
        self._mark_state()

    def partial(self, text: str):
        if self.ctx is None:
            return
        self.ctx.words = tuple(self.backend.question_ids(text))
        if self.retrieve_steps is not None:
            self.ctx.want = text                # a Retrieve job takes it when the GPU loop gets to it
            return
        alt = self.retrieve_alt(text) if self.retrieve_alt is not None else None
        self._apply_retrieval(text, self.retrieve(text), alt)

    def _apply_retrieval(self, text: str, pages, alt):
        self._set_pages(pages)
        if self.retrieve_alt is not None or (self.retrieve_steps is not None and hasattr(self.backend, "exact_len")):
            self.ctx.alt = alt if alt and tuple(alt) != self.ctx.pages else None
        self._diag("page_log", "partial", text, _label(self.ctx.pages), self.ctx.frontier, _label(self.ctx.alt))

    def mic_silent(self):
        if self.state is State.LISTENING:
            self.state = State.TAIL
            self._mark_state()

    def speech(self):
        if self.state is State.TAIL:
            self.state = State.LISTENING
            self._mark_state()

    def _diag(self, name: str, *row):
        """Diagnostics (M3): a timestamped row in this turn's log `name`."""
        if self.turn is not None:
            self.turn.setdefault(name, []).append((round(self.clock() - self.turn["t_down"], 3), *row))

    def _mark_state(self):
        if self.turn is not None:
            self.turn["states"].append((round(self.clock() - self.turn["t_down"], 3), self.state.name))

    def key_up(self, t: Optional[float] = None):
        if self.ctx is None:
            return
        self.state = State.RELEASED
        self._mark_state()
        need = self.newest_frame if self.freshness == "B" else -1
        self.answer = Answer(need)
        self._add(self.answer)
        self.turn.update(t_up=self.clock() if t is None else t, need_frame=need,
                         screen_at_up=self.screen, waited_for_read=need > self.screen)
        self._sync_frontier()
        self.turn.update(frontier_at_up=self.ctx.frontier, pages_at_up=_label(self.ctx.pages),
                         alt_ready_at_up=(self.backend.exact_len(self.ctx.alt) if self.ctx.alt and
                                          hasattr(self.backend, "exact_len") else None))

    def final(self, text: str):
        if self.ctx is None:
            return
        self.ctx.words = tuple(self.backend.question_ids(text))
        self.ctx.final = True
        self.ctx.alt = None
        self.ctx.want = None
        for j in [j for j in self.jobs if isinstance(j, Retrieve)]:
            self._drop(j)
        self._set_pages(self.retrieve(text))
        self._diag("page_log", "final", text, _label(self.ctx.pages), self.ctx.frontier, None)

    def cancel_turn(self):
        """The turn ends without an answer (nothing was said, speech recognition failed)."""
        if self.ctx is None:
            return
        if self.answer is not None:
            self._drop(self.answer)
        self._close_turn(cancelled=True)

    def frame(self, fid: int):
        """A changed screen to read. Latest frame wins, except over a read an answer waits for. A frame no newer
        than one already seen is ignored: replacing a newer read with it could leave an answer waiting forever."""
        if fid <= self.newest_frame:
            self.stats["frames_ignored"] = self.stats.get("frames_ignored", 0) + 1
            return
        self.newest_frame = fid
        if self.read is not None:
            if self._pinned(self.read):
                self.pending_frame = fid
                return
            self._drop(self.read)
            self.stats["reads_dropped"] += 1
        self._start_read(fid)

    def visit_ended(self, page):
        """A page visit finished: its stored piece and the timeline need (re)building."""
        self.add_build(page)
        self.prefix_dirty = True

    def add_build(self, item):
        """Background work for one item (a page's stored piece, a kept screen's keys): run in quiet time, or during a
        turn if that turn's context needs it."""
        if (self._buildable(item) and not self.backend.has_piece(item)
                and not any(isinstance(j, Build) and j.page == item for j in self.jobs)):
            self._add(Build(self, item))

    def _buildable(self, item) -> bool:
        f = getattr(self.backend, "buildable", None)
        return True if f is None else f(item)

    # ------------------------------------------------------------ the loop

    def step(self) -> bool:
        """Run one piece of the best runnable job. False if nothing can run."""
        self._sync_frontier()
        for j in [j for j in self.jobs if j.gen is not None and j.stale(self)]:
            self._drop(j)
        self._ensure_jobs()
        job = self._pick()
        if job is None:
            return False
        self._run(job)
        return True

    def key(self) -> tuple:
        c = self.ctx
        return (self.prefix, c.pages, c.frontier, self.screen)

    def missing(self) -> list:
        c = self.ctx
        return [p for p in c.pages[c.frontier:] if not self.backend.has_piece(p)]

    # ------------------------------------------------------------ internals

    def _add(self, job: Job):
        self._seq += 1
        job.seq = self._seq
        self.jobs.append(job)

    def _drop(self, job: Job):
        if isinstance(job, Prefix):
            self.prefix_dirty = True                # still needed, with the newer history
        if job.gen is not None:
            job.gen.close()
        if job in self.jobs:
            self.jobs.remove(job)
        self.stats["wasted_s"] += job.gpu_s
        if self.turn is not None:
            self.turn["wasted_s"] += job.gpu_s
        if job is self.read:
            self.read = None
        if job is self.answer:
            self.answer = None

    def _pinned(self, r: Read) -> bool:
        return self.answer is not None and r.fid <= self.answer.need

    def _start_read(self, fid: int):
        self.read = Read(self, fid)
        self._add(self.read)

    def _read_done(self, r: Read):
        self.screen = max(self.screen, r.fid)
        self.read = None
        if self.pending_frame is not None:
            fid, self.pending_frame = self.pending_frame, None
            self._start_read(fid)

    def _set_pages(self, pages):
        c = self.ctx
        pages = tuple(pages)
        c.stats["retrievals"] += 1
        if pages == c.pages:
            return
        i = 0
        while i < min(len(pages), len(c.pages)) and pages[i] == c.pages[i]:
            i += 1
        c.stats["page_changes"] += 1
        c.stats["pages_cut"] += max(0, c.frontier - i)
        c.frontier = min(c.frontier, i)
        c.pages = pages
        self.backend.prefetch(pages)
        for p in pages:
            self.add_build(p)

    def _sync_frontier(self):
        """With a branching backend, the frontier is whatever an exact branch holds for the current pages: a late switch
        to a context computed ahead (the runner-up) starts there, and an evicted branch can't be assumed."""
        f = getattr(self.backend, "exact_len", None)
        c = self.ctx
        if f is not None and c is not None and self.prefix == getattr(self.backend, "prefix_v", self.prefix):
            n = f(c.pages)
            if n > c.frontier and self.turn is not None:
                self.turn["frontier_found"] = max(self.turn.get("frontier_found", 0), n - c.frontier)
            c.frontier = n

    def _prefix_changed(self):
        """A new prefix: everything computed after the old one is invalid."""
        self.prefix += 1
        if self.ctx is not None:
            self.ctx.stats["pages_cut"] += self.ctx.frontier
            self.ctx.frontier = 0

    def _near_up(self) -> bool:
        """TAIL, or key up with the final transcript not in yet (~40 ms; the answer can't start before it). Work for
        the current context runs then in short pieces (TAIL's limit): it is kept if the final transcript keeps the
        context, dropped as stale if not (M3, 2026-11-15)."""
        return self.state is State.TAIL or (self.state is State.RELEASED and self.ctx is not None
                                            and not self.ctx.final)

    def _spec_phase(self) -> bool:
        c = self.ctx
        return c is not None and (self._near_up() or
                                  (self.state is State.LISTENING and c.frontier == len(c.pages)))

    def _question_reusable(self, key, ids) -> int:
        """How many of `ids` a finished question prefill already covers for this context."""
        sp = self.ctx.spec
        if sp is None or sp[0] != key or ids[:len(sp[1])] != sp[1]:
            return 0
        return len(sp[1])

    def _ensure_jobs(self):
        if (self.prefix_dirty and (self.state is State.QUIET or self.prefix == 0)
                and not any(isinstance(j, Prefix) for j in self.jobs)):
            self._add(Prefix(self))                 # created when it can run, so it takes the history as it is
            self.prefix_dirty = False
        if self.prefix > 0 and self.ex_in < self.ex_done and not any(isinstance(j, Append) for j in self.jobs):
            self._add(Append(self))
        c = self.ctx
        if c is None or not (self.state is State.LISTENING or self._near_up()):
            return
        if c.want is not None and not c.final and not any(isinstance(j, Retrieve) for j in self.jobs):
            text, c.want = c.want, None
            self._add(Retrieve(self, text))
        if (c.frontier < len(c.pages) and (self.state is State.LISTENING or self.missing())
                and not any(isinstance(j, Upgrade) for j in self.jobs)):
            self._add(Upgrade(self))
        if (c.words and self._spec_phase() and not self.missing() and c.spec != (self.key(), c.words)
                and not any(isinstance(j, Spec) for j in self.jobs)):
            self._add(Spec(self))
        if (self.state is State.LISTENING and c.alt and c.frontier == len(c.pages)
                and self.backend.exact_len(c.alt) < len(c.alt) and not any(isinstance(j, AltUpgrade) for j in self.jobs)):
            self._add(AltUpgrade(self))

    def _pick(self) -> Optional[Job]:
        limit, allowed = self.limits[self.state], ALLOWED[self.state]
        if self.state is State.RELEASED and self._near_up():     # before the final transcript: see _near_up
            limit, allowed = self.limits[State.TAIL], allowed | {Cls.IMPROVING}
        best, best_k = None, None
        for j in self.jobs:
            c = j.cls(self)
            if c not in allowed or not j.runnable(self):
                continue
            if c >= Cls.IMPROVING and self.est.get(j.next_kind) > limit:
                self.est.blocked(j.next_kind)
                if self.turn is not None:
                    b = self.turn["blocked"].setdefault(f"{self.state.name}:{type(j).__name__}:{j.next_kind}",
                                                        {"n": 0, "est_ms": 0.0})
                    b["n"] += 1
                    b["est_ms"] = round(1000 * self.est.get(j.next_kind), 1)
                continue
            k = (c, j.order, j.seq)
            if best_k is None or k < best_k:
                best, best_k = j, k
        return best

    def _plan(self) -> dict:
        c = self.ctx
        key = self.key()
        have = self._question_reusable(key, c.words)
        return {"key": key, "pages": c.pages, "frontier": c.frontier, "screen": self.screen, "words": c.words,
                "base_reused": c.spec is not None and c.spec[0] == key, "question_reused": have,
                "prefill": len(c.words) - have, "missing": self.missing(),
                "on_first_token": self._first_token}

    def _run(self, job: Job):
        state, cls = self.state, job.cls(self)
        t0 = self.clock()
        if isinstance(job, Answer) and job.gen is None:
            job.plan = self._plan()
            job.prime(self.backend.answer(job.plan))
        kind = job.next_kind
        est = self.est.t.get(kind)
        done, result = False, None
        job.ran = True
        try:
            job.next_kind = next(job.gen)
        except StopIteration as e:
            done, result = True, e.value
        except Exception as e:                          # one failed job must not stop the loop (§3.7's rule)
            import traceback
            self.stats["errors"] = self.stats.get("errors", 0) + 1
            self.errors = (self.errors + [(type(job).__name__, kind, traceback.format_exc())])[-20:]
            if self.turn is not None:                   # diagnostics: a job that keeps failing shows in the turn
                self.turn.setdefault("job_errors", []).append(f"{type(job).__name__}:{kind}: {e!r}"[:200])
            self._drop(job)
            if isinstance(job, Answer) and self.turn is not None:
                self.turn["error"] = f"{type(job).__name__} failed in {kind}: {e!r}"
                self._close_turn(cancelled=True)
            return
        dt = self.clock() - t0
        self.est.update(kind, dt)
        job.gpu_s += dt
        self.stats["pieces"] += 1
        self.log.append((t0, dt, state, cls, kind, est))
        if self.turn is not None:
            self.turn["gpu_s"][cls.name] += dt
            k = f"{state.name}:{cls.name}:{kind}"
            self.turn["pieces"][k] = round(self.turn["pieces"].get(k, 0.0) + dt, 4)
        if done:
            if job in self.jobs:
                self.jobs.remove(job)
            job.finish(self, result)

    def _first_token(self):
        if self.turn is not None and "t_first" not in self.turn:
            self.turn["t_first"] = self.clock()
            self.turn["release_s"] = self.turn["t_first"] - self.turn["t_up"]
        if self.state is State.RELEASED:
            self.state = State.ANSWERING
            self._mark_state()

    def _answer_done(self, a: Answer, result: Optional[dict] = None):
        self.answer = None
        result = result or {}
        if result.get("exchange"):
            self.ex_done += 1
        if result.get("rebuild"):                       # e.g. the history outgrew its cap: drop the oldest
            self.prefix_dirty = True
        self._close_turn(cancelled=False, plan=a.plan)

    def _close_turn(self, cancelled: bool, plan: Optional[dict] = None):
        if self.turn is not None:
            c = self.ctx
            self.turn.update(cancelled=cancelled, **c.stats)
            if plan is not None:
                self.turn["plan"] = {k: v for k, v in plan.items() if k != "on_first_token"}
            self.turns.append(self.turn)
            self.turns = self.turns[-200:]
            if self.on_turn is not None:
                self.on_turn(self.turn)
        self.turn, self.ctx, self.state = None, None, State.QUIET


# ---------------------------------------------------------------- simulation (tests, design checks)

class FakeClock:
    def __init__(self, t: float = 0.0):
        self.t = t

    def __call__(self) -> float:
        return self.t


class SimBackend:
    """GPU work replaced by fixed piece times on a fake clock. Piece kinds and the counts per job follow the real
    work: a screen read is prep, patch embedding, 16 vision layers, pooling, embeddings and 12 two-layer text
    pieces (S0, §5.8); page work and question prefill run in layer pieces."""

    def __init__(self, clock: FakeClock, times: dict[str, float], built=(), page_pieces: int = 24,
                 question_pieces: int = 6, answer_tokens: int = 10, history: bool = True):
        self.clock, self.priors, self.history = clock, dict(times), history
        self.built = set(built)
        self.page_pieces, self.question_pieces, self.answer_tokens = page_pieces, question_pieces, answer_tokens
        self.calls: list[tuple] = []
        self.branches: list[tuple] = []         # exact caches: the slots each holds, most recent first (max 3)
        self.branch_v = 0                       # prefix version the branches were built on
        self.prefix_v = 0                       # prefix version the backend holds
        self.spec_state: tuple = (None, ())     # (key, ids) of the last finished question prefill
        self.ex_recorded = 0                    # exchanges answered
        self.frames: dict = {}                  # the worker hands frames over here, as to the real backend
        import threading
        self.frames_lock = threading.Lock()
        self.prefix_ex = 0                      # exchanges in the prefix the backend holds

    def _pieces(self, kinds):
        for k in kinds:
            yield k
            self.clock.t += self.priors[k]

    def read(self, fid):
        self.calls.append(("read", fid))
        return self._pieces(["prep", "patch"] + ["vision"] * 16 + ["pool", "embed"] + ["text"] * 12)

    def build(self, page):
        def g():
            self.calls.append(("build", page))
            yield from self._pieces(["page_layer"] * self.page_pieces)
            self.built.add(page)
        return g()

    def prefix(self, n_exchanges):
        def g():
            self.calls.append(("prefix", n_exchanges))
            yield from self._pieces(["page_layer"] * self.page_pieces)
            assert n_exchanges <= self.ex_recorded
            self.prefix_ex = n_exchanges                       # a new prefix: nothing after it is valid
            self.prefix_v += 1
        return g()

    def append(self, i):
        def g():
            self.calls.append(("append", i))
            yield "q_layers"
            assert i == self.prefix_ex < self.ex_recorded, (i, self.prefix_ex, self.ex_recorded)
            self.clock.t += self.priors["q_layers"]
            yield from self._pieces(["q_layers"] * (self.question_pieces - 1))
            self.prefix_ex = i + 1
            self.prefix_v += 1
        return g()

    def _branches(self) -> list:
        if self.branch_v != self.prefix_v:
            self.branches, self.branch_v = [], self.prefix_v
        return self.branches

    def exact_len(self, pages) -> int:
        pages = tuple(pages)
        best = 0
        for b in self._branches():
            n = 0
            while n < min(len(b), len(pages)) and b[n] == pages[n]:
                n += 1
            best = max(best, n)
        return best

    def upgrade(self, pages, i):
        """Page i on top of a branch holding pages[:i], as RecallBackend: checked when the work starts, recorded as a
        new branch when it ends; at most 3 branches, least recently used out."""
        def g():
            self.calls.append(("upgrade", pages, i))
            yield "page_layer"
            assert i == 0 or self.exact_len(pages) >= i, (self.branches, pages, i)
            v = self.prefix_v
            self.clock.t += self.priors["page_layer"]
            yield from self._pieces(["page_layer"] * (self.page_pieces - 1))
            if v == self.prefix_v:
                bs = self._branches()
                if tuple(pages[:i]) in bs:
                    bs.remove(tuple(pages[:i]))
                bs.insert(0, tuple(pages[:i + 1]))
                del bs[3:]
        return g()

    def spec(self, key, ids, have):
        def g():
            self.calls.append(("spec", key, ids, have))
            yield "q_layers"
            assert have == 0 or (self.spec_state[0] == key and ids[:have] == self.spec_state[1][:have])
            self.clock.t += self.priors["q_layers"]
            yield from self._pieces(["q_layers"] * (self.question_pieces - 1))
            self.spec_state = (key, ids)
        return g()

    def answer(self, plan):
        def g():
            yield "assemble"
            f = plan["frontier"]
            assert f == 0 or (self.exact_len(plan["pages"]) >= f and self.prefix_v == plan["key"][0]), \
                (self.branches, plan["pages"], f, self.prefix_v, plan["key"])
            assert self.prefix_ex == self.ex_recorded, (self.prefix_ex, self.ex_recorded)   # history complete
            if plan["question_reused"]:
                k, ids = self.spec_state
                assert k == plan["key"] and plan["words"][:plan["question_reused"]] == ids, (self.spec_state, plan)
            self.clock.t += self.priors["assemble"]
            for _ in plan["missing"]:
                for _ in range(self.page_pieces):
                    yield "page_layer"
                    self.clock.t += self.priors["page_layer"]
            if plan["prefill"]:
                for _ in range(self.question_pieces):
                    yield "q_layers"
                    self.clock.t += self.priors["q_layers"]
            plan["on_first_token"]()
            for _ in range(self.answer_tokens):
                yield "decode"
                self.clock.t += self.priors["decode"]
            self.ex_recorded += self.history
            return {"exchange": self.history, "rebuild": False}
        return g()

    def has_piece(self, page) -> bool:
        return page in self.built

    def prefetch(self, pages):
        self.calls.append(("prefetch", tuple(pages)))

    def question_ids(self, text):
        return tuple(text.split())


def simulate(s: Scheduler, events: list[tuple], until: Optional[float] = None):
    """events: (t, method, *args). Runs pieces between events; an event that arrives during a piece is handled
    after it, as on the GPU, but key_down / key_up keep their own time for latency."""
    clock = s.clock
    ev = sorted(events, key=lambda e: e[0])
    i = 0
    while until is None or clock.t < until:
        while i < len(ev) and ev[i][0] <= clock.t:
            t, name, *args = ev[i]
            fn = getattr(s, name)
            fn(*args, t=t) if name in ("key_down", "key_up") else fn(*args)
            i += 1
        if s.step():
            continue
        if i >= len(ev):
            break
        clock.t = ev[i][0]
