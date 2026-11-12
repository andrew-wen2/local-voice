"""Turn scheduler (scheduler.py, ARCHITECTURE.md §5.9) on a fake clock. Piece times: S0 for screen reads (§5.8);
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
    return all(est <= PIECE_LIMIT_S[st] + 1e-9 for _, _, st, c, _, est in s.log if c >= Cls.IMPROVING)


def classes_allowed(s):
    return all(c in ALLOWED[st] for _, _, st, c, _, _ in s.log)


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


def test_no_tail_means_question_prefilled_at_release():
    s, _, _ = make()
    simulate(s, turn(silent=None, up=0.6))
    p = s.turns[-1]["plan"]
    assert p["question_reused"] == 0 and p["prefill"] == 4
    assert s.turns[-1]["release_s"] == pytest.approx(0.03 + T["assemble"] + 6 * T["q_layers"])


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
