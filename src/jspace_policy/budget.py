from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

# Modal public rates checked 2026-08-11.
GPU_USD_PER_SECOND = {
    "T4": 0.000164,
    "L4": 0.000222,
    "A10": 0.000306,
    "L40S": 0.000542,
    "A100-40GB": 0.000583,
    "A100-80GB": 0.000694,
    "H100": 0.001097,
}
CPU_CORE_USD_PER_SECOND = 0.0000131
MEMORY_GIB_USD_PER_SECOND = 0.00000222


@dataclass(frozen=True)
class ExecutionLimit:
    """A hard Modal function timeout and its billable resources."""

    timeout_seconds: int
    gpu: str | None
    cpu_cores: float
    memory_gib: float


# Operational limits only: the frozen RBG-5B dataset, behavioral gates, patch
# search, and locked endpoint are unchanged. Centralizing the limits makes the
# user's incremental authorization executable and testable.
# The initial authorization was $10.00.  After the first behavior run and the
# timed-out discovery attempt, the user reported $6.60 remaining on 2026-08-31.
# These limits cover only the prospective repair run: discovery -> locked ->
# J-space.  The failed attempt is historical spend, not silently omitted from
# the authorization boundary.
RBG5B_INCREMENTAL_COST_LIMIT_USD = 10.0
RBG5B_REPAIR_REMAINING_COST_LIMIT_USD = 6.6
RBG5B_SALVAGE_REMAINING_COST_LIMIT_USD = 2.5
RBG5B_EXECUTION_LIMITS = {
    "preflight": ExecutionLimit(300, None, 2, 8),
    "behavior": ExecutionLimit(1200, "A100-80GB", 8, 32),
    "discovery": ExecutionLimit(1200, "A100-80GB", 16, 64),
    "locked": ExecutionLimit(3000, "A100-80GB", 16, 64),
    "jspace": ExecutionLimit(720, "A100-80GB", 8, 64),
}

RBG5B_REPAIR_EXECUTION_LIMITS = {
    stage: RBG5B_EXECUTION_LIMITS[stage]
    for stage in ("discovery", "locked", "jspace")
}

# After locked run 33380084999 spent its full 3,000-second timeout on serial
# CPU bootstrap work while retaining an A100, the user authorized one bounded
# implementation-only salvage.  The frozen dataset, site, endpoints, and
# resampling count do not change; only resource placement and vectorization do.
RBG5B_SALVAGE_EXECUTION_LIMITS = {
    "locked_gpu": ExecutionLimit(1200, "A100-80GB", 16, 64),
    "jspace": ExecutionLimit(720, "A100-80GB", 8, 64),
}


@dataclass(frozen=True)
class CostEstimate:
    gpu: str | None
    seconds: float
    gpu_usd: float
    cpu_usd: float
    memory_usd: float
    subtotal_usd: float
    buffered_usd: float


def execution_limit_cost_usd(limit: ExecutionLimit) -> float:
    """Return the maximum compute charge implied by one hard timeout."""

    gpu_rate = 0.0 if limit.gpu is None else GPU_USD_PER_SECOND[limit.gpu]
    rate = (
        gpu_rate
        + CPU_CORE_USD_PER_SECOND * limit.cpu_cores
        + MEMORY_GIB_USD_PER_SECOND * limit.memory_gib
    )
    return rate * limit.timeout_seconds


def execution_plan_cost_usd(limits: dict[str, ExecutionLimit]) -> float:
    """Return the cost ceiling if every stage consumes its full timeout."""

    return sum(execution_limit_cost_usd(limit) for limit in limits.values())


def admit_execution_plan(
    limits: dict[str, ExecutionLimit], *, limit_usd: float
) -> float:
    """Refuse a plan whose hard timeout ceiling exceeds its authorization."""

    ceiling = execution_plan_cost_usd(limits)
    if ceiling > limit_usd:
        raise RuntimeError(
            f"execution plan refused: hard timeout ceiling ${ceiling:.2f} "
            f"exceeds ${limit_usd:.2f} authorization"
        )
    return ceiling


def estimate_cost(
    gpu: str | None,
    seconds: float,
    *,
    cpu_cores: float = 4.0,
    memory_gib: float = 16.0,
    uncertainty_fraction: float = 0.20,
    gpu_count: int = 1,
) -> CostEstimate:
    if gpu is not None and gpu not in GPU_USD_PER_SECOND:
        raise ValueError(f"unknown GPU: {gpu}")
    if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 1:
        raise ValueError("gpu_count must be a positive integer")
    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    gpu_cost = 0.0 if gpu is None else GPU_USD_PER_SECOND[gpu] * seconds * gpu_count
    cpu_cost = CPU_CORE_USD_PER_SECOND * cpu_cores * seconds
    memory_cost = MEMORY_GIB_USD_PER_SECOND * memory_gib * seconds
    subtotal = gpu_cost + cpu_cost + memory_cost
    return CostEstimate(
        gpu=gpu,
        seconds=seconds,
        gpu_usd=gpu_cost,
        cpu_usd=cpu_cost,
        memory_usd=memory_cost,
        subtotal_usd=subtotal,
        buffered_usd=subtotal * (1 + uncertainty_fraction),
    )


def ledger_total(path: Path) -> float:
    if not path.exists():
        return 0.0
    total = 0.0
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            total += float(json.loads(line).get("buffered_usd", 0.0))
    return total


def admit_run(path: Path, estimate: CostEstimate, *, study_limit_usd: float = 25.0) -> None:
    projected = ledger_total(path) + estimate.buffered_usd
    if projected > study_limit_usd:
        raise RuntimeError(
            f"run refused: conservative ledger would reach ${projected:.2f}, "
            f"above the ${study_limit_usd:.2f} study limit"
        )


def append_ledger(
    path: Path,
    estimate: CostEstimate,
    *,
    run_id: str,
    stage: str,
    status: str = "measured",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "run_id": run_id,
        "stage": stage,
        "status": status,
        **asdict(estimate),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
