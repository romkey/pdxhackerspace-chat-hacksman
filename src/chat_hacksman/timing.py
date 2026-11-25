"""How long each stage of answering a chat took.

A StageTimer records one chat's stages in the order they ran; Storage keeps
the running tally per stage (record_timings, get_timing_stats).
"""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from time import perf_counter

# Stage names, in the order a chat runs them.
REWRITE = "rewrite"
EMBED = "embed"
QDRANT = "qdrant"
RERANK = "rerank"
SELECT = "select"
EVENTS = "events"
LLM = "llm"
# Time to first token: part of llm, not added to it.
TTFT = "first_token"
FINISH = "finish"
TOTAL = "total"
STAGE_ORDER = (REWRITE, EMBED, QDRANT, RERANK, SELECT, EVENTS, LLM, TTFT, FINISH, TOTAL)


def stage_sort_key(name: str) -> tuple[int, str]:
    """Known stages in run order; `qdrant:<collection>` right after `qdrant`;
    unknown ones just before `total`."""
    base = name.split(":", 1)[0]
    rank = STAGE_ORDER.index(base) if base in STAGE_ORDER else len(STAGE_ORDER) - 1
    return (rank, name)


@dataclass(slots=True)
class StageTimer:
    """Milliseconds per stage for one chat; a stage timed twice adds up."""

    started: float = field(default_factory=perf_counter)
    stages: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, ms: float) -> None:
        self.stages[name] = self.stages.get(name, 0.0) + ms

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time the block as `name`, even if it raises."""
        begun = perf_counter()
        try:
            yield
        finally:
            self.add(name, (perf_counter() - begun) * 1000.0)

    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started) * 1000.0

    def report(self) -> dict[str, float]:
        """Each stage rounded to 0.01 ms, in run order, with `total` (wall
        time so far) last."""
        names = sorted((name for name in self.stages if name != TOTAL), key=stage_sort_key)
        out = {name: round(self.stages[name], 2) for name in names}
        out[TOTAL] = round(self.elapsed_ms(), 2)
        return out
