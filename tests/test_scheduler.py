"""Turn scheduler (scheduler.py) on a fake clock. Piece times: measured for screen reads;
page and question work in layer pieces (estimates, not measured)."""
import pytest

from scheduler import (ALLOWED, PIECE_LIMIT_S, Cls, FakeClock, Scheduler, SimBackend, State, simulate)

T = {"prep": .061, "patch": .088, "vision": .080, "pool": .010, "embed": .008, "text": .042,
     "page_layer": .020, "q_layers": .007, "assemble": .005, "decode": .026}
READ_S = .061 + .088 + 16 * .080 + .010 + .008 + 12 * .042


def make(retrieve=None, built=("a", "b", "c", "d"), freshness="B", **kw):
    clock = FakeClock()
    be = SimBackend(clock, T, built=built, **kw)
    s = Scheduler(be, retrieve or (lambda text: ("a", "b")), clock=clock, freshness=freshness)
    s.frame(0)                                   # the screen at startup
    simulate(s, [])
    assert s.screen == 0 and s.prefix == 1       # first read and first prefix build
    clock.t = 10.0
    return s, be, clock


def turn(t0=10.0, words="what was the price", up=1.3, silent=1.0, final_lag=0.03, partials=((0.3, None),)):
    ev = [(t0, "key_down")]
    for dt, text in partials:
        ev.append((t0 + dt, "partial", text or words))
    if silent is not None:
        ev.append((t0 + silent, "mic_silent"))
    ev += [(t0 + up, "key_up"), (t0 + up + final_lag, "final", words)]
    return ev


def limits_held(s):
    """Every IMPROVING / BACKGROUND piece fit its state's limit when it started."""
    lim = {**PIECE_LIMIT_S, State.RELEASED: PIECE_LIMIT_S[State.TAIL]}     # before the final transcript (_near_up)
    return all(est <= lim[st] + 1e-9 for _, _, st, c, _, est in s.log if c >= Cls.IMPROVING)


def classes_allowed(s):
    # IMPROVING runs after key up only until the final transcript is in (scheduler._near_up)
    return all(c in ALLOWED[st] or (st is State.RELEASED and c is Cls.IMPROVING) for _, _, st, c, _, _ in s.log)


def test_still_screen_turn_speculation_covers_the_question():
    s, be, _ = make()
    simulate(s, turn())
    t = s.turns[-1]
    p = t["plan"]
    # page a exact (0.3 + 24 * 20 ms < 1.0); page b was being upgraded when the mic went quiet: frozen
    assert p["frontier"] == 1 and p["pages"] == ("a", "b")
    assert p["base_reused"] and p["question_reused"] == 4 and p["prefill"] == 0
    # final transcript (30 ms after key up) + assembly; no question prefill left
    assert t["release_s"] == pytest.approx(0.03 + T["assemble"])
    assert limits_held(s) and classes_allowed(s)


def test_no_tail_means_question_prefilled_while_the_final_transcript_comes():
    """With no quiet tail, the question prefill starts at key up, in the ~30 ms before the final
    transcript, and finishes before the answer once the final transcript keeps it; it isn't redone in the answer.
    Before: all of it after the final transcript, 0.03 + assemble + 6 question pieces."""
    s, _, _ = make()
    simulate(s, turn(silent=None, up=0.6))
    t = s.turns[-1]
    p = t["plan"]
    assert p["question_reused"] == 4 and p["prefill"] == 0
    assert t["pieces"]["RELEASED:IMPROVING:q_layers"] > 0 and t["pieces"]["RELEASED:CRITICAL:q_layers"] > 0
    assert t["release_s"] < 0.03 + T["assemble"] + 6 * T["q_layers"] - 0.005
    assert limits_held(s) and classes_allowed(s)


def test_a_prefill_the_final_transcript_changes_is_dropped():
    """The same, but the final transcript differs from the last partial: the started prefill is stale, the answer
    prefills the final words itself."""
    s, _, _ = make()
    simulate(s, turn(silent=None, up=0.6, words="what was the cost", partials=((0.3, "what was the price"),)))
    p = s.turns[-1]["plan"]
    assert p["question_reused"] == 0 and p["prefill"] == 4 and p["words"][-1] == "cost"


def test_retrieval_change_cuts_the_exact_frontier():
    pages = {"q1": ("a", "b", "c"), "q2": ("a", "d", "c")}
    s, be, _ = make(retrieve=lambda text: pages[text.split()[0]])
    ev = [(10.0, "key_down"), (10.05, "partial", "q1 x"), (11.2, "partial", "q2 x y"),
          (11.5, "mic_silent"), (11.8, "key_up"), (11.83, "final", "q2 x y")]
    simulate(s, ev)
    t = s.turns[-1]
    assert t["pages_cut"] == 1                   # a and b were exact; b left the set
    upgrades = [c for c in be.calls if c[0] == "upgrade"]
    assert ("upgrade", ("a", "d"), 1) in upgrades
    assert t["plan"]["pages"] == ("a", "d", "c")
    assert t["plan"]["question_reused"] == 3     # speculation in TAIL over the new set


def test_screen_change_during_question_blocks_answer_under_B():
    s, _, _ = make()
    ev = turn(up=1.2) + [(11.0, "frame", 1)]
    simulate(s, ev)
    t = s.turns[-1]
    assert t["waited_for_read"] and t["plan"]["screen"] == 1
    assert t["release_s"] > READ_S - 0.2 - 0.3   # most of the read is left at key up


def test_rule_C_answers_from_the_last_read():
    s, _, _ = make(freshness="C")
    simulate(s, turn(up=1.2) + [(11.0, "frame", 1)])
    t = s.turns[-1]
    assert not t["waited_for_read"] and t["plan"]["screen"] == 0 and t["release_s"] < 0.1


def test_frame_after_key_up_is_not_waited_for_and_doesnt_drop_the_needed_read():
    s, be, _ = make()
    simulate(s, turn(up=1.2) + [(11.0, "frame", 1), (11.3, "frame", 2)])
    t = s.turns[-1]
    assert t["plan"]["screen"] == 1 and s.stats["reads_dropped"] == 0
    assert s.screen == 2                         # read after the answer


def test_an_older_frame_never_replaces_a_newer_read():
    s, _, _ = make()
    simulate(s, turn(up=1.2) + [(10.5, "frame", 3), (10.9, "frame", 1)])
    assert s.turns[-1]["plan"]["screen"] == 3 and s.stats["frames_ignored"] == 1


def test_latest_frame_wins_in_quiet():
    s, _, _ = make()
    simulate(s, [(10.0, "frame", 1), (10.5, "frame", 2)])
    assert s.stats["reads_dropped"] == 1 and s.screen == 2


def test_background_waits_for_quiet_and_timeline_is_fixed_for_the_turn():
    s, be, clock = make()
    simulate(s, turn() + [(10.5, "visit_ended", "e")])
    t = s.turns[-1]
    assert t["prefix_at_down"] == 1 and t["plan"]["key"][0] == 1
    assert all(c is not Cls.BACKGROUND for _, _, st, c, _, _ in s.log if st is not State.QUIET)
    assert "e" in be.built
    assert ("prefix", 1) in be.calls             # rebuilt after the answer, with its exchange
    assert s.prefix == 3 and s.ex_in == s.ex_done == 1


def test_missing_piece_is_built_during_the_turn():
    s, be, _ = make(retrieve=lambda text: ("a", "e"), built=("a",))
    simulate(s, turn(up=2.0, silent=1.8))
    p = s.turns[-1]["plan"]
    assert "e" in be.built and p["missing"] == []
    assert be.calls.index(("build", "e")) < be.calls.index(("upgrade", ("a",), 0))   # before the upgrade


def test_missing_piece_at_key_up_is_computed_by_the_answer():
    s, be, _ = make(retrieve=lambda text: ("a", "e"), built=("a",))
    simulate(s, turn(up=0.35, silent=None))
    assert s.turns[-1]["plan"]["missing"] == ["e"]


def test_new_words_extend_the_question_prefill():
    s, be, _ = make()
    ev = turn(words="what was the price", partials=((0.3, "what was the"),), silent=1.0, up=1.3)
    simulate(s, ev)
    p = s.turns[-1]["plan"]
    assert p["question_reused"] == 3 and p["prefill"] == 1


def test_speech_after_tail_resumes_the_upgrade():
    s, be, _ = make()
    ev = turn(silent=0.6, up=2.0) + [(10.7, "speech")]
    simulate(s, ev)
    p = s.turns[-1]["plan"]
    assert p["frontier"] == 2                    # finished after speech resumed
    assert p["question_reused"] == 4             # redone over the complete context


def test_barge_in_cancels_the_answer():
    s, be, _ = make()
    be.answer_tokens = 200
    ev = turn() + [(11.6, "key_down"), (11.9, "partial", "and the date"), (12.2, "key_up"),
                   (12.23, "final", "and the date")]
    simulate(s, ev)
    assert [t["cancelled"] for t in s.turns] == [True, False]
    assert s.state is State.QUIET and s.ctx is None


def test_unknown_piece_kind_raises():
    s, be, _ = make()
    be.priors.pop("page_layer")
    s.est.t.pop("page_layer")
    with pytest.raises(KeyError):
        simulate(s, turn())


@pytest.mark.parametrize("seed", range(300))
def test_random_turns_keep_the_invariants(seed):
    """Random speech timing, partial transcripts, retrieval changes, screen changes and page visits. The sim
    backend asserts that whatever an answer claims (exact pages, a finished question prefill) is what it holds."""
    import random
    rng = random.Random(seed)
    pool = list("abcdefgh")
    sets = {w: tuple(sorted(rng.sample(pool, rng.randint(1, 4)))) for w in ("p", "q", "r")}
    s, be, clock = make(retrieve=lambda text: sets[text.split()[-1][0]], built=rng.sample(pool, 5),
                        freshness=rng.choice("BC"))
    be.answer_tokens = rng.randint(1, 40)
    ev, t, frame = [], 10.0, 1
    for _ in range(rng.randint(1, 4)):
        up = rng.uniform(0.2, 3.0)
        ev.append((t, "key_down"))
        k = rng.randint(0, 4)
        times = sorted(rng.uniform(0, up) for _ in range(k))
        words = []
        for x in times:
            words.append(rng.choice(["p", "q", "r"]) + str(len(words)))
            ev.append((t + x, "partial", " ".join(words)))
        if rng.random() < 0.6:
            q = rng.uniform(0, up)
            ev.append((t + q, "mic_silent"))
            if rng.random() < 0.3:
                ev.append((t + rng.uniform(q, up), "speech"))
        if not words or rng.random() < 0.5:          # the final transcript adds a word, or matches the last partial
            words.append(rng.choice(["p", "q", "r"]) + "f")
        final = " ".join(words)
        ev += [(t + up, "key_up"), (t + up + rng.uniform(0.01, 0.1), "final", final)]
        for x in sorted(rng.uniform(-1, up + 1) for _ in range(rng.randint(0, 3))):
            ev.append((t + x, "frame", frame))        # ids in time order, as the capture assigns them
            frame += 1
        for _ in range(rng.randint(0, 2)):
            ev.append((t + rng.uniform(-1, up + 1), "visit_ended", rng.choice(pool)))
        t += up + rng.uniform(0.1, 4.0)              # some next questions barge in on the answer
    simulate(s, ev)
    assert s.state is State.QUIET and s.ctx is None and s.answer is None
    assert len(s.turns) == sum(1 for e in ev if e[1] == "key_down")
    assert limits_held(s) and classes_allowed(s)
    for tr in s.turns:
        if tr["cancelled"]:
            continue
        p = tr["plan"]
        assert p["pages"] == sets[p["words"][-1][0]] if p["words"] else True
        assert 0 <= p["frontier"] <= len(p["pages"]) and p["prefill"] + p["question_reused"] == len(p["words"])
        assert p["screen"] >= tr["need_frame"]       # rule B: never older than the newest frame at key up
        assert p["key"][0] >= tr["prefix_at_down"]
        assert p["key"][0] > 0
        assert not (set(p["missing"]) & set(p["pages"][:p["frontier"]]))


def test_a_failing_answer_closes_its_turn_and_the_loop_goes_on():
    s, be, _ = make()
    real = be.answer

    def boom(plan):
        def g():
            yield "assemble"
            raise ValueError("window layers would attend to stale entries")
        return g()
    be.answer = boom
    simulate(s, turn())
    t = s.turns[-1]
    assert t["cancelled"] and "ValueError" in t["error"] and s.state is State.QUIET and s.answer is None
    be.answer = real
    simulate(s, turn(t0=20.0))
    assert s.turns[-1]["cancelled"] is False and s.stats["errors"] == 1


def test_runner_up_context_is_ready_when_retrieval_switches_late():
    """With time to spare during speech, the runner-up's slots are computed exactly too, so a switch
    to them in the last partial finds every slot exact; without the runner-up, the switch starts from nothing."""
    first, runner = ("s1", "op", "v1"), ("s2", "op", "v2")

    def run(alt):
        clock = FakeClock()
        be = SimBackend(clock, T, built=first + runner, page_pieces=8)      # a context: 3 x 8 x 20 ms
        s = Scheduler(be, lambda text: runner if "late" in text else first, clock=clock,
                      retrieve_alt=(lambda text: None if "late" in text else runner) if alt else None)
        s.frame(0)
        simulate(s, [])
        clock.t = 10.0
        ev = [(10.0, "key_down"), (10.1, "partial", "what time is"), (12.2, "partial", "what time is late"),
              (12.3, "key_up"), (12.33, "final", "what time is late")]
        simulate(s, ev)
        return s, be

    s, be = run(alt=True)
    t = s.turns[-1]
    assert t["plan"]["pages"] == runner and t["plan"]["frontier"] == 3 and not t["plan"]["missing"]
    assert ("upgrade", runner, 0) in be.calls and t.get("frontier_found") == 3
    assert limits_held(s) and classes_allowed(s)
    s, be = run(alt=False)
    assert s.turns[-1]["plan"]["frontier"] == 0      # the switch threw the exact work away


def test_runner_up_waits_for_the_current_context_and_its_question():
    """The runner-up is computed only after the current context is exact, and never instead of its question prefill."""
    s, be, _ = make(retrieve=lambda text: ("a", "b"))
    s.retrieve_alt = lambda text: ("a", "c")
    simulate(s, turn(up=2.5, silent=None))
    calls = [c for c in be.calls if c[0] in ("upgrade", "spec")]
    first_alt = next(i for i, c in enumerate(calls) if c[0] == "upgrade" and c[1] == ("a", "c"))
    assert ("upgrade", ("a", "b"), 1) in calls[:first_alt] and any(c[0] == "spec" for c in calls[:first_alt])
    assert s.turns[-1]["plan"]["frontier"] == 2


def test_retrieval_runs_in_pieces_and_takes_the_newest_partial():
    """Retrieval is a job of GPU pieces, not a blocking call in partial(). Partials that arrive while
    one runs aren't retrieved; the newest is, next. The final transcript is still retrieved before the answer."""
    clock = FakeClock()
    be = SimBackend(clock, {**T, "rq": .03}, built=("a", "b", "c"))
    asked = []

    def pages(text):
        return ("a", "b") if "price" in text else ("a", "c")

    def steps(text):
        for _ in range(3):
            yield "rq"
            clock.t += .03
        asked.append(text)
        return pages(text), None

    s = Scheduler(be, pages, clock=clock, retrieve_steps=steps)
    s.frame(0)
    simulate(s, [])
    clock.t = 10.0
    ev = [(10.0, "key_down"), (10.30, "partial", "what"), (10.32, "partial", "what was"),
          (10.34, "partial", "what was the"), (10.36, "partial", "what was the price"),
          (11.5, "key_up"), (11.53, "final", "what was the price")]
    simulate(s, ev)
    assert asked == ["what", "what was the price"]
    assert s.turns[-1]["plan"]["pages"] == ("a", "b") and limits_held(s) and classes_allowed(s)
    assert any(k.endswith(":rq") for k in s.turns[-1]["pieces"])


def test_page_build_continues_in_tail_when_the_question_prefill_cannot_run():
    """With no stored pieces for the pages, the question prefill can't run in TAIL; building the
    pages then is the only useful work, so it isn't frozen. With stored pieces, TAIL still prefers the prefill."""
    s, be, _ = make(retrieve=lambda text: ("x", "y"), built=(), page_pieces=8)
    be.buildable = lambda page: False           # like a kept screen or its view: no stored piece can be made
    simulate(s, turn(up=1.2, silent=0.35, partials=((0.3, None),)))
    t = s.turns[-1]
    assert t["plan"]["frontier"] == 2 and limits_held(s) and classes_allowed(s)
    assert t["pieces"].get("TAIL:IMPROVING:page_layer", 0) > 0


def test_retrieval_waiting_on_the_disk_lets_page_work_run():
    """While a retrieval's next piece waits for keys to come off the disk, the job isn't runnable
    (it would otherwise hold the GPU loop: other work runs meanwhile), and it runs once the read is in. (The fake
    clock jumps to the next event when nothing can run, so the read is seen as done only at the next event.)"""
    clock = FakeClock()
    be = SimBackend(clock, {**T, "rq": .005}, built=("a", "b", "c"), page_pieces=8)
    io = {"done_at": 10.5}
    be.retrieval_ready = lambda: clock.t >= io["done_at"]
    asked = []

    def steps(text):
        yield "rq"
        clock.t += .005
        asked.append(text)
        return ("a", "b"), None

    s = Scheduler(be, lambda text: ("a", "b"), clock=clock, retrieve_steps=steps)
    s.frame(0)
    simulate(s, [])
    clock.t = 10.0
    simulate(s, [(10.0, "key_down"), (10.05, "partial", "what"), (10.06, "partial", "what was the price"),
                 (11.5, "key_up"), (11.53, "final", "what was the price")])
    assert asked == ["what", "what was the price"] and limits_held(s) and classes_allowed(s)
    rq_t = [t0 for t0, _, _, _, kind, _ in s.log if kind == "rq"]
    assert rq_t and min(rq_t) >= 10.5                       # not before its read was in


def test_one_slow_piece_does_not_hold_its_kind_back_for_good():
    """One outlier doesn't move the estimate; repeated slow pieces push it over the limit, and then
    the kind is retried after losing BLOCKS_TO_FORGET picks (it can't be re-measured otherwise)."""
    from scheduler import Estimates
    e = Estimates({"rs": 0.005})
    e.update("rs", 0.19)
    assert e.get("rs") == 0.005                      # one outlier: the lower median of prior and sample
    e.update("rs", 0.06)
    assert e.get("rs") == 0.06                       # two of three slow: over a 50 ms limit
    for _ in range(Estimates.BLOCKS_TO_FORGET):
        e.blocked("rs")
    assert e.get("rs") == 0.005                      # the largest forgotten: it runs again and is re-measured
    e.update("rs", 0.004)
    assert e.get("rs") <= 0.005
