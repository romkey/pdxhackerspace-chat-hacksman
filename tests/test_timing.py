import pytest

from chat_hacksman.timing import StageTimer


def test_stage_timer_adds_repeated_stages_and_keeps_run_order() -> None:
    timer = StageTimer()
    timer.add("llm", 10.0)
    timer.add("qdrant:hs_member", 1.0)
    timer.add("qdrant", 1.0)
    timer.add("qdrant", 2.004)
    timer.add("rewrite", 3.0)
    timer.add("mystery", 4.0)

    report = timer.report()

    assert list(report) == [
        "rewrite",
        "qdrant",
        "qdrant:hs_member",
        "llm",
        "mystery",
        "total",
    ]
    assert report["qdrant"] == 3.0
    assert report["total"] >= 0.0


def test_stage_timer_times_a_block_that_raises() -> None:
    timer = StageTimer()
    with pytest.raises(RuntimeError), timer.stage("llm"):
        raise RuntimeError("down")
    assert timer.stages["llm"] >= 0.0
