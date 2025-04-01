import os 
from opentelemetry import trace  
from opentelemetry.sdk.trace.sampling import Decision, Sampler, SamplingResult
from typing import Sequence, Dict, Optional, Sequence
import random
from opentelemetry.trace import Link, SpanKind
from opentelemetry.util.types import Attributes
from opentelemetry.context import Context
from threading import Lock
from enum import Enum, auto
import time

class QueryTraceSampler(Sampler):
    """
    Samples traces based on a ratio, making the decision at spans named 'start'.
    All children of a 'start' span inherit its sampling decision.
    Other spans inherit their parent's decision. Root spans are always sampled initially.
    """

    def __init__(self, ratio: float):
        print(f"Initializing QueryTraceSampler with ratio: {ratio}")
        if not 0.0 <= ratio <= 1.0:
            raise ValueError("Sampling ratio must be between 0.0 and 1.0")
        self._ratio = ratio
        self._bound = int(ratio * (2**64 - 1)) # Precompute bound for comparison

    def should_sample(
        self,
        parent_context: Optional[Context],
        trace_id: int,
        name: str,
        kind: SpanKind = None,
        attributes: Attributes = None,
        links: Sequence[Link] = None,
        trace_state = None,
    ) -> SamplingResult:
        
        decision = Decision.DROP # Default decision

        # Check parent sampling decision first
        parent_span_context = trace.get_current_span(parent_context).get_span_context()

        # --- Decision Logic ---
        if name.endswith(" start"): # Check if the name ENDS with " start"
            if random.random() < self._ratio:
                 print(f"QueryTraceSampler: RECORDING span '{name}' (Random < {self._ratio})")
                 decision = Decision.RECORD_AND_SAMPLE
            else:
                 print(f"QueryTraceSampler: DROPPING span '{name}' (Random >= {self._ratio})")
                 decision = Decision.DROP

        elif parent_span_context is not None and parent_span_context.is_valid:
             if parent_span_context.trace_flags.sampled:
                decision = Decision.RECORD_AND_SAMPLE
             else:
                decision = Decision.DROP
        else:
            decision = Decision.RECORD_AND_SAMPLE # Record root spans


        # Ensure trace_state is propagated if present
        resulting_trace_state = trace_state or parent_span_context.trace_state

        return SamplingResult(
            decision,
            attributes, # Attributes are not modified by this sampler
            resulting_trace_state,
        )

    def get_description(self) -> str:
        return f"QueryTraceSampler{{ratio={self._ratio}}}"


class ScheduleDecision(Enum):
    """Decision returned by the PyTorchStyleScheduler."""
    SKIP = auto()    # Do not trace this step.
    WARMUP = auto()  # Do not trace this step (warming up).
    RECORD = auto()  # Trace and record this step.
    DONE = auto()    # All scheduled cycles are complete.

class Scheduler:
    """
    Determines whether a trace should be initiated based on a schedule
    conceptually similar to torch.profiler.schedule.

    Operates on step counts (e.g., queries, batches).

    Args:
        wait (int): Number of initial steps to skip.
        warmup (int): Number of steps to treat as warmup after waiting
                      (these steps are *not* traced).
        active (int): Number of steps to trace and record after warmup.
        repeat (int): Number of times to repeat the (warmup, active) cycle
                      after the initial (wait, warmup, active) cycle.
                      repeat=0 means one cycle total.
                      repeat=1 means two cycles total.
    """
    def __init__(self, wait: int = 1, warmup: int = 1, active: int = 3, repeat: int = 0):
        if not all(isinstance(i, int) and i >= 0 for i in [wait, warmup, active, repeat]):
            raise ValueError("wait, warmup, active, and repeat must be non-negative integers.")
        if active <= 0:
            # Ensure there's at least one active step if we are to record anything
            print("Warning: schedule active count is 0. No steps will be recorded.")
            # Allow it, but it won't do much unless warmup/wait are non-zero

        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat # Number of *additional* cycles

        self._lock = Lock()
        self._step_count = 0 # Total steps processed since start
        self._total_cycles = 1 + self.repeat
        self._steps_per_cycle = self.warmup + self.active

        print(f"Initializing PyTorchStyleScheduler:")
        print(f"  Wait: {self.wait}")
        print(f"  Warmup: {self.warmup}")
        print(f"  Active: {self.active}")
        print(f"  Repeat: {self.repeat} (Total cycles: {self._total_cycles})")
        if self._steps_per_cycle <= 0 and self.wait <= 0:
             print("Warning: Scheduler configured with no wait, warmup, or active steps.")

    def step(self) -> ScheduleDecision:
        """
        Advances the scheduler state by one step and returns the decision
        for the *current* step.
        """
        with self._lock:
            self._step_count += 1
            current_step = self._step_count

            # 1. Check if we are in the initial wait phase
            if current_step <= self.wait:
                # print(f"  Step {current_step}: WAITING") # Debug
                return ScheduleDecision.SKIP

            # 2. Calculate position relative to the start of cycles
            steps_after_wait = current_step - self.wait

            # Check if cycles are possible
            if self._steps_per_cycle <= 0:
                 # print(f"  Step {current_step}: DONE (no steps per cycle)") # Debug
                 return ScheduleDecision.DONE # No warmup/active steps defined

            # 3. Determine the current cycle index (0-based)
            current_cycle_idx = (steps_after_wait - 1) // self._steps_per_cycle

            # 4. Check if all requested cycles are completed
            if current_cycle_idx >= self._total_cycles:
                # print(f"  Step {current_step}: DONE (cycle {current_cycle_idx} >= {self._total_cycles})") # Debug
                return ScheduleDecision.DONE

            # 5. Determine the step number within the current cycle (1-based)
            step_in_cycle = (steps_after_wait - 1) % self._steps_per_cycle + 1

            # 6. Check if we are in the warmup phase of the current cycle
            if step_in_cycle <= self.warmup:
                # print(f"  Step {current_step}: WARMUP (cycle {current_cycle_idx}, step_in_cycle {step_in_cycle})") # Debug
                return ScheduleDecision.WARMUP
            # 7. Otherwise, we must be in the active phase of the current cycle
            else:
                # print(f"  Step {current_step}: ACTIVE (cycle {current_cycle_idx}, step_in_cycle {step_in_cycle})") # Debug
                return ScheduleDecision.RECORD

    def get_step_count(self) -> int:
         with self._lock:
             return self._step_count
