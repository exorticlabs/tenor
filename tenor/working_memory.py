"""
Executive Working Memory Engine, Task State & Execution Bus (working_memory.py)
==============================================================================

Implements Layer 4: Volatile executive memory and task orchestration for autonomous agents.
Unifies:
- Complete tool execution telemetry (ToolRunRecord with parameters, duration, exit codes, and side effects).
- Hierarchical task and subtask lifecycle management (TaskRecord, SubTask, TaskProgress, TaskPhase).
- Real-time background job registry (BackgroundTaskRecord with live PIDs, status, and log streaming).
- Structured scratchpad (hypotheses, verified observations, active roadblocks, next actions).
- Dynamic workspace path resolution & dual-tier cache anchoring (task-scoped vs turn-TTL).
- Dynamic working attention mask calculation for long-term EEKG retrieval biasing.
- Slotted Quota working context block generation for deterministic prompt injection.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# ============================================================================
# Status & Phase Enumerations
# ============================================================================

class TaskStatus(str, Enum):
    """Execution status for subtasks and steps in the DAG."""
    PENDING = "PENDING"
    IN_PROGRESS = "IN_PROGRESS"
    COMPLETED = "COMPLETED"
    BLOCKED = "BLOCKED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TaskPhase(str, Enum):
    """High-level lifecycle phase of an active agent task."""
    IDLE = "IDLE"
    WORKSPACE_RESOLVING = "WORKSPACE_RESOLVING"
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    WAITING_USER = "WAITING_USER"
    BACKGROUND_BUSY = "BACKGROUND_BUSY"
    BLOCKED = "BLOCKED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    ABORTED = "ABORTED"


class ToolExecutionStatus(str, Enum):
    """Status of an individual tool call execution."""
    PENDING = "PENDING"
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    BACKGROUNDED = "BACKGROUNDED"
    CANCELLED = "CANCELLED"


# ============================================================================
# Data Models: Tool Runs, SubTasks, Tasks & Background Jobs
# ============================================================================

@dataclass
class ToolRunRecord:
    """Complete operational telemetry for an individual tool invocation."""
    call_id: str
    step_number: int
    tool_name: str
    arguments: Dict[str, Any]
    status: ToolExecutionStatus = ToolExecutionStatus.PENDING
    task_id: str = ""
    subtask_id: Optional[str] = None
    intent: str = ""
    exit_code: Optional[int] = None
    stdout_summary: str = ""
    raw_output_ref: Optional[str] = None
    files_read: List[str] = field(default_factory=list)
    files_written: List[str] = field(default_factory=list)
    files_deleted: List[str] = field(default_factory=list)
    duration_ms: float = 0.0
    timestamp: float = field(default_factory=time.time)
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "call_id": self.call_id,
            "step_number": self.step_number,
            "tool_name": self.tool_name,
            "arguments": dict(self.arguments),
            "status": self.status.value,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "intent": self.intent,
            "exit_code": self.exit_code,
            "stdout_summary": self.stdout_summary,
            "raw_output_ref": self.raw_output_ref,
            "files_read": list(self.files_read),
            "files_written": list(self.files_written),
            "files_deleted": list(self.files_deleted),
            "duration_ms": self.duration_ms,
            "timestamp": self.timestamp,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ToolRunRecord:
        status_val = d.get("status", ToolExecutionStatus.PENDING)
        if isinstance(status_val, str):
            try:
                status_enum = ToolExecutionStatus(status_val)
            except ValueError:
                status_enum = ToolExecutionStatus.PENDING
        else:
            status_enum = status_val

        return cls(
            call_id=d.get("call_id", ""),
            step_number=int(d.get("step_number", 0)),
            tool_name=d.get("tool_name", ""),
            arguments=dict(d.get("arguments", {})),
            status=status_enum,
            task_id=d.get("task_id", ""),
            subtask_id=d.get("subtask_id"),
            intent=d.get("intent", ""),
            exit_code=d.get("exit_code"),
            stdout_summary=d.get("stdout_summary", ""),
            raw_output_ref=d.get("raw_output_ref"),
            files_read=list(d.get("files_read", [])),
            files_written=list(d.get("files_written", [])),
            files_deleted=list(d.get("files_deleted", [])),
            duration_ms=float(d.get("duration_ms", 0.0)),
            timestamp=float(d.get("timestamp", time.time())),
            error_message=d.get("error_message"),
        )


@dataclass
class SubTask:
    """Atomic step in the task DAG with dependency pointers and entity bindings."""
    task_id: str
    description: str
    dependencies: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    status: TaskStatus = TaskStatus.PENDING
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result_summary: Optional[str] = None
    milestone: Optional[str] = None
    intent: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "dependencies": list(self.dependencies),
            "entities": list(self.entities),
            "status": self.status.value,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
            "result_summary": self.result_summary,
            "milestone": self.milestone,
            "intent": self.intent,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SubTask:
        status_val = d.get("status", TaskStatus.PENDING)
        if isinstance(status_val, str):
            try:
                status_enum = TaskStatus(status_val)
            except ValueError:
                status_enum = TaskStatus.PENDING
        else:
            status_enum = status_val

        return cls(
            task_id=d.get("task_id", ""),
            description=d.get("description", ""),
            dependencies=list(d.get("dependencies", [])),
            entities=list(d.get("entities", [])),
            status=status_enum,
            created_at=float(d.get("created_at", time.time())),
            completed_at=d.get("completed_at"),
            result_summary=d.get("result_summary"),
            milestone=d.get("milestone"),
            intent=d.get("intent"),
        )


@dataclass
class TaskProgress:
    """Dynamic progress tracking and milestone metrics."""
    total_subtasks: int = 0
    completed_subtasks: int = 0
    pending_subtasks: int = 0
    blocked_subtasks: int = 0
    percent_complete: float = 0.0
    current_milestone: str = ""
    active_blocker: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_subtasks": self.total_subtasks,
            "completed_subtasks": self.completed_subtasks,
            "pending_subtasks": self.pending_subtasks,
            "blocked_subtasks": self.blocked_subtasks,
            "percent_complete": self.percent_complete,
            "current_milestone": self.current_milestone,
            "active_blocker": self.active_blocker,
        }


@dataclass
class BackgroundTaskRecord:
    """Tracking descriptor for asynchronous terminal jobs and builds."""
    job_id: str
    task_id: str
    command: str
    cwd: str
    pid: Optional[int] = None
    status: str = "RUNNING"  # RUNNING | COMPLETED | FAILED | CANCELLED
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    exit_code: Optional[int] = None
    log_tail: str = ""
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_id": self.task_id,
            "command": self.command,
            "cwd": self.cwd,
            "pid": self.pid,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "exit_code": self.exit_code,
            "log_tail": self.log_tail,
            "summary": self.summary,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> BackgroundTaskRecord:
        return cls(
            job_id=d.get("job_id", ""),
            task_id=d.get("task_id", ""),
            command=d.get("command", ""),
            cwd=d.get("cwd", ""),
            pid=d.get("pid"),
            status=d.get("status", "RUNNING"),
            start_time=float(d.get("start_time", time.time())),
            end_time=d.get("end_time"),
            exit_code=d.get("exit_code"),
            log_tail=d.get("log_tail", ""),
            summary=d.get("summary", ""),
        )


@dataclass
class TaskRecord:
    """
    Comprehensive task execution container:
    Holds parameters, requirements, acceptance criteria, pinned workspace,
    and historical tool run context.
    """
    task_id: str
    goal: str
    phase: TaskPhase = TaskPhase.IDLE
    status: TaskStatus = TaskStatus.IN_PROGRESS
    workspace_path: str = ""
    task_parameters: Dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: List[str] = field(default_factory=list)
    user_constraints: List[str] = field(default_factory=list)
    progress: TaskProgress = field(default_factory=TaskProgress)
    tool_history: List[ToolRunRecord] = field(default_factory=list)
    active_background_jobs: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result_summary: Optional[str] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "goal": self.goal,
            "phase": self.phase.value,
            "workspace_path": self.workspace_path,
            "task_parameters": dict(self.task_parameters),
            "acceptance_criteria": list(self.acceptance_criteria),
            "user_constraints": list(self.user_constraints),
            "progress": self.progress.to_dict(),
            "tool_history": [t.to_dict() for t in self.tool_history],
            "active_background_jobs": list(self.active_background_jobs),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "completed_at": self.completed_at,
            "result_summary": self.result_summary,
            "error_message": self.error_message,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TaskRecord:
        phase_val = d.get("phase", TaskPhase.IDLE)
        if isinstance(phase_val, str):
            try:
                phase_enum = TaskPhase(phase_val)
            except ValueError:
                phase_enum = TaskPhase.IDLE
        else:
            phase_enum = phase_val

        tools = [ToolRunRecord.from_dict(t) for t in d.get("tool_history", []) if isinstance(t, dict)]
        prog_d = d.get("progress", {})
        progress = TaskProgress(
            total_subtasks=prog_d.get("total_subtasks", 0),
            completed_subtasks=prog_d.get("completed_subtasks", 0),
            pending_subtasks=prog_d.get("pending_subtasks", 0),
            blocked_subtasks=prog_d.get("blocked_subtasks", 0),
            percent_complete=prog_d.get("percent_complete", 0.0),
            current_milestone=prog_d.get("current_milestone", ""),
            active_blocker=prog_d.get("active_blocker"),
        )

        return cls(
            task_id=d.get("task_id", ""),
            goal=d.get("goal", ""),
            phase=phase_enum,
            workspace_path=d.get("workspace_path", ""),
            task_parameters=dict(d.get("task_parameters", {})),
            acceptance_criteria=list(d.get("acceptance_criteria", [])),
            user_constraints=list(d.get("user_constraints", [])),
            progress=progress,
            tool_history=tools,
            active_background_jobs=list(d.get("active_background_jobs", [])),
            created_at=float(d.get("created_at", time.time())),
            updated_at=float(d.get("updated_at", time.time())),
            completed_at=d.get("completed_at"),
            result_summary=d.get("result_summary"),
            error_message=d.get("error_message"),
        )


# ============================================================================
# Structured Scratchpad
# ============================================================================

@dataclass
class StructuredScratchpad:
    """
    Volatile scratchpad with dedicated semantic compartments:
    - Hypotheses: Active theories, diagnoses, and architectural approaches.
    - Observations: Concrete empirical facts discovered from tool outputs.
    - Roadblocks: Active hurdles, errors, or constraints requiring pivot.
    - Next Actions: Immediate, concrete next 1-3 planned steps.
    """
    hypotheses: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    roadblocks: List[str] = field(default_factory=list)
    next_actions: List[str] = field(default_factory=list)

    def add_hypothesis(self, text: str) -> None:
        clean = str(text).strip()
        if clean and clean not in self.hypotheses:
            self.hypotheses.append(clean)
            if len(self.hypotheses) > 5:
                self.hypotheses.pop(0)

    def add_observation(self, text: str) -> None:
        clean = str(text).strip()
        if clean and clean not in self.observations:
            self.observations.append(clean)
            if len(self.observations) > 10:
                self.observations.pop(0)

    def add_roadblock(self, text: str) -> None:
        clean = str(text).strip()
        if clean and clean not in self.roadblocks:
            self.roadblocks.append(clean)
            if len(self.roadblocks) > 5:
                self.roadblocks.pop(0)

    def clear_roadblocks(self) -> None:
        self.roadblocks.clear()

    def add_next_action(self, text: str) -> None:
        clean = str(text).strip()
        if clean and clean not in self.next_actions:
            self.next_actions.append(clean)
            if len(self.next_actions) > 5:
                self.next_actions.pop(0)

    def clear_next_actions(self) -> None:
        self.next_actions.clear()

    def to_dict(self) -> Dict[str, List[str]]:
        return {
            "hypotheses": list(self.hypotheses),
            "observations": list(self.observations),
            "roadblocks": list(self.roadblocks),
            "next_actions": list(self.next_actions),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StructuredScratchpad:
        return cls(
            hypotheses=list(d.get("hypotheses", [])),
            observations=list(d.get("observations", [])),
            roadblocks=list(d.get("roadblocks", [])),
            next_actions=list(d.get("next_actions", [])),
        )


# ============================================================================
# Executive Working Memory Engine
# ============================================================================

class WorkingMemoryEngine:
    """
    Layer 4: Executive Working Memory Engine and Task Execution Bus.
    
    Responsibilities:
    - Maintains active goal, task DAG, subtasks, and atomic topological progression.
    - Tracks full operational tool telemetry (ToolRunRecord with parameters, stdout, exit codes).
    - Manages live background task registry (BackgroundTaskRecord with live PIDs and log tails).
    - Houses the StructuredScratchpad (hypotheses, observations, roadblocks, next actions).
    - Governs dual-tier workspace path caching (task-scoped vs turn-TTL) and agent override.
    - Computes volatile attention seeds biasing the long-term 4D Spatio-Temporal Tensor.
    - Yields high-density Slotted Quota markdown blocks for prompt injection.
    """

    def __init__(self, root_goal: str = "") -> None:
        self.root_goal: str = root_goal
        self.subtasks: Dict[str, SubTask] = {}
        self.current_subtask_id: Optional[str] = None
        self.active_files: Set[str] = set()
        self.active_variables: Dict[str, Any] = {}
        self.terminal_registers: Dict[str, Any] = {}
        self.active_error: Optional[Dict[str, Any]] = None
        self.working_entities: Set[str] = set()

        # Enterprise Task & Execution State
        self.active_task: Optional[TaskRecord] = None
        self.tasks: Dict[str, TaskRecord] = {}
        self.tool_history: List[ToolRunRecord] = []
        self.background_tasks: Dict[str, BackgroundTaskRecord] = {}
        self.scratchpad: StructuredScratchpad = StructuredScratchpad()

        # Workspace Path Anchoring & Caching
        self.active_workspace_path: str = ""
        self.workspace_cache_scope: str = "task"  # "task" | "turn"
        self.workspace_cache_ttl: int = 0

        if self.root_goal:
            self._refresh_working_entities()

    # ------------------------------------------------------------------------
    # Core Goal & SubTask DAG (100% Backward Compatible)
    # ------------------------------------------------------------------------

    def set_root_goal(self, goal: str) -> None:
        """Sets or updates the primary agent objective."""
        self.root_goal = goal.strip()
        if self.active_task and not self.active_task.goal:
            self.active_task.goal = self.root_goal
        self._refresh_working_entities()

    def add_subtask(
        self,
        task_id: str,
        description: str,
        dependencies: Optional[List[str]] = None,
        entities: Optional[List[str]] = None,
        milestone: Optional[str] = None,
        intent: Optional[str] = None,
    ) -> SubTask:
        """
        Adds a new atomic task to the DAG. If idle and all dependencies are met,
        transitions task immediately to IN_PROGRESS.
        """
        deps = list(dependencies or [])
        ents = list(entities or [])

        task = SubTask(
            task_id=task_id,
            description=description,
            dependencies=deps,
            entities=ents,
            status=TaskStatus.PENDING,
            milestone=milestone,
            intent=intent,
        )
        self.subtasks[task_id] = task

        # Auto-schedule if idle and ready
        if self.current_subtask_id is None:
            all_deps_done = all(
                self.subtasks.get(d) and self.subtasks[d].status == TaskStatus.COMPLETED
                for d in deps
            )
            if all_deps_done:
                task.status = TaskStatus.IN_PROGRESS
                self.current_subtask_id = task_id

        self.recalculate_progress()
        self._refresh_working_entities()


        return task

    def transition_subtask(
        self,
        task_id: str,
        new_status: TaskStatus,
        result_summary: Optional[str] = None,
    ) -> None:
        """
        Transitions status of task_id. On completion, automatically schedules
        the next ready subtask in DAG topological order.
        """
        task = self.subtasks.get(task_id)
        if not task:
            logger.warning(f"Subtask '{task_id}' not found for transition.")
            return

        task.status = new_status
        if new_status == TaskStatus.COMPLETED:
            task.completed_at = time.time()
            if result_summary:
                task.result_summary = result_summary
            if self.current_subtask_id == task_id:
                self.current_subtask_id = None

        # Topological progression: find next ready PENDING task
        if self.current_subtask_id is None:
            for candidate_id, candidate in self.subtasks.items():
                if candidate.status == TaskStatus.PENDING:
                    deps_satisfied = all(
                        self.subtasks.get(d) and self.subtasks[d].status == TaskStatus.COMPLETED
                        for d in candidate.dependencies
                    )
                    if deps_satisfied:
                        candidate.status = TaskStatus.IN_PROGRESS
                        self.current_subtask_id = candidate_id
                        break

        self.recalculate_progress()
        self._refresh_working_entities()

    def complete_subtask(self, task_id: str, result_summary: Optional[str] = None) -> None:
        """Marks subtask as COMPLETED and triggers topological DAG progression."""
        self.transition_subtask(task_id, TaskStatus.COMPLETED, result_summary=result_summary)

    def fail_subtask(self, task_id: str, error: Optional[str] = None) -> None:
        """Marks subtask as FAILED and sets active error blocker."""
        self.transition_subtask(task_id, TaskStatus.FAILED, result_summary=error)
        if error:
            self.record_error(error_type="SubTaskFailure", message=error)

    def get_ready_subtasks(self) -> List[SubTask]:
        """Returns all subtasks that are ready for execution (dependencies satisfied)."""
        ready = []
        for candidate_id, candidate in self.subtasks.items():
            if candidate.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS):
                deps_satisfied = all(
                    self.subtasks.get(d) and self.subtasks[d].status == TaskStatus.COMPLETED
                    for d in candidate.dependencies
                )
                if deps_satisfied:
                    ready.append(candidate)
        return ready

    # ------------------------------------------------------------------------
    # Task Management & Lifecycle (Enterprise Layer)
    # ------------------------------------------------------------------------

    def create_task(
        self,
        goal: str,
        workspace_path: str = "",
        task_id: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        acceptance_criteria: Optional[List[str]] = None,
        user_constraints: Optional[List[str]] = None,
    ) -> TaskRecord:
        """Initializes a new top-level TaskRecord and binds it as active."""
        tid = task_id or f"task_{int(time.time() * 1000)}"
        clean_goal = str(goal).strip()

        record = TaskRecord(
            task_id=tid,
            goal=clean_goal,
            phase=TaskPhase.EXECUTING,
            workspace_path=str(workspace_path).strip(),
            task_parameters=dict(parameters or {}),
            acceptance_criteria=list(acceptance_criteria or []),
            user_constraints=list(user_constraints or []),
        )

        self.active_task = record
        self.tasks[tid] = record
        self.root_goal = clean_goal

        if workspace_path:
            self.set_workspace_path(workspace_path, scope="task")

        self.recalculate_progress()
        self._refresh_working_entities()
        return record

    def get_active_task(self) -> Optional[TaskRecord]:
        """Returns the currently active task record, if one exists."""
        return self.active_task

    def set_task_phase(self, phase: TaskPhase) -> None:
        """Updates the operational phase of the active task."""
        if self.active_task:
            self.active_task.phase = phase
            self.active_task.updated_at = time.time()

    def update_progress(
        self,
        milestone: Optional[str] = None,
        blocker: Optional[str] = None,
    ) -> TaskProgress:
        """Explicitly sets active milestone or blocker on task progress."""
        progress = self.recalculate_progress()
        if milestone is not None:
            progress.current_milestone = milestone
        if blocker is not None:
            progress.active_blocker = blocker
        if self.active_task:
            self.active_task.progress = progress
            self.active_task.updated_at = time.time()
        return progress

    def recalculate_progress(self) -> TaskProgress:
        """Recalculates progress metrics based on DAG subtasks."""
        total = len(self.subtasks)
        completed = sum(1 for t in self.subtasks.values() if t.status == TaskStatus.COMPLETED)
        blocked = sum(1 for t in self.subtasks.values() if t.status == TaskStatus.BLOCKED)
        pending = sum(1 for t in self.subtasks.values() if t.status in (TaskStatus.PENDING, TaskStatus.IN_PROGRESS))
        pct = (completed / total * 100.0) if total > 0 else 0.0

        current_milestone = ""
        if self.current_subtask_id and self.current_subtask_id in self.subtasks:
            curr = self.subtasks[self.current_subtask_id]
            current_milestone = curr.milestone or curr.description

        active_blocker = None
        if self.active_error:
            active_blocker = f"{self.active_error.get('type')}: {self.active_error.get('message', '')[:120]}"

        progress = TaskProgress(
            total_subtasks=total,
            completed_subtasks=completed,
            pending_subtasks=pending,
            blocked_subtasks=blocked,
            percent_complete=round(pct, 1),
            current_milestone=current_milestone,
            active_blocker=active_blocker,
        )

        if self.active_task:
            self.active_task.progress = progress
            self.active_task.updated_at = time.time()

        return progress

    def complete_task(self, result_summary: Optional[str] = None) -> None:
        """Marks the active task as completed."""
        if not self.active_task:
            return
        self.active_task.phase = TaskPhase.COMPLETED
        self.active_task.completed_at = time.time()
        self.active_task.updated_at = time.time()
        if result_summary:
            self.active_task.result_summary = result_summary
        self.recalculate_progress()
        self._refresh_working_entities()

    def fail_task(self, reason: str) -> None:
        """Marks the active task as failed with an error message."""
        if not self.active_task:
            return
        self.active_task.phase = TaskPhase.FAILED
        self.active_task.completed_at = time.time()
        self.active_task.updated_at = time.time()
        self.active_task.error_message = reason
        self.recalculate_progress()
        self._refresh_working_entities()

    def abort_task(self, reason: str = "User aborted") -> None:
        """Aborts the active task, halting execution."""
        if not self.active_task:
            return
        self.active_task.phase = TaskPhase.ABORTED
        self.active_task.completed_at = time.time()
        self.active_task.updated_at = time.time()
        self.active_task.error_message = reason
        self.recalculate_progress()
        self._refresh_working_entities()

    # ------------------------------------------------------------------------
    # Workspace Resolution & Dual-Tier Caching
    # ------------------------------------------------------------------------

    def set_workspace_path(self, path: str, scope: str = "task", ttl_turns: int = 5, reason: str = "") -> None:
        """
        Anchors the canonical project workspace path.
        Scope:
        - "task": Persists for the full duration of the active task.
        - "turn": Persists for `ttl_turns` conversational turns before expiring.
        """
        clean = str(path).strip()
        if not clean:
            return

        norm = os.path.abspath(clean)
        self.active_workspace_path = norm
        self.workspace_cache_scope = scope
        self.workspace_cache_ttl = ttl_turns if scope == "turn" else 999999

        if self.active_task:
            self.active_task.workspace_path = norm
            self.active_task.updated_at = time.time()

    def get_workspace_path(self) -> str:
        """Returns the active workspace path if valid and not expired."""
        if self.workspace_cache_scope == "turn" and self.workspace_cache_ttl <= 0:
            return ""
        return self.active_workspace_path

    def tick_workspace_cache(self) -> None:
        """Decrements TTL for turn-scoped workspace cache."""
        if self.workspace_cache_scope == "turn" and self.workspace_cache_ttl > 0:
            self.workspace_cache_ttl -= 1
            if self.workspace_cache_ttl <= 0:
                self.active_workspace_path = ""

    decrement_turn_ttls = tick_workspace_cache

    # ------------------------------------------------------------------------
    # Tool Execution Telemetry Bus
    # ------------------------------------------------------------------------

    def record_tool_run(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        status: Any = ToolExecutionStatus.SUCCESS,
        output: Any = None,
        intent: str = "",
        exit_code: Optional[int] = None,
        stdout_summary: str = "",
        raw_output_ref: Optional[str] = None,
        files_read: Optional[List[str]] = None,
        files_written: Optional[List[str]] = None,
        files_deleted: Optional[List[str]] = None,
        duration_ms: float = 0.0,
        error_message: Optional[str] = None,
        touched_files: Optional[List[str]] = None,
    ) -> ToolRunRecord:
        """
        Records full operational telemetry for a tool execution step.
        Automatically updates active files, errors, and task history.
        """
        if isinstance(status, str):
            s_upper = status.strip().upper()
            if s_upper in ("COMPLETED", "SUCCESS", "OK"):
                status_enum = ToolExecutionStatus.SUCCESS
            elif s_upper in ("FAILED", "ERROR", "CRASH"):
                status_enum = ToolExecutionStatus.ERROR
            elif s_upper == "TIMEOUT":
                status_enum = ToolExecutionStatus.TIMEOUT
            elif s_upper == "BACKGROUNDED":
                status_enum = ToolExecutionStatus.BACKGROUNDED
            else:
                try:
                    status_enum = ToolExecutionStatus(s_upper)
                except ValueError:
                    status_enum = ToolExecutionStatus.SUCCESS
        else:
            status_enum = status

        if not stdout_summary and output is not None:
            if isinstance(output, dict):
                stdout_summary = output.get("stdout") or output.get("output") or output.get("message") or json.dumps(output, ensure_ascii=False)
            else:
                stdout_summary = str(output)
            stdout_summary = stdout_summary.strip()
            if len(stdout_summary) > 250:
                stdout_summary = stdout_summary[:250] + "..."

        step_num = len(self.tool_history) + 1
        tid = self.active_task.task_id if self.active_task else ""
        sub_id = self.current_subtask_id

        f_read = list(files_read or [])
        f_write = list(files_written or [])
        f_del = list(files_deleted or [])
        if touched_files:
            f_write.extend(touched_files)

        # Auto-track touched files
        for f in f_read + f_write:
            self.record_file_touch(f)

        # Auto-track errors if execution failed
        if status_enum in (ToolExecutionStatus.ERROR, ToolExecutionStatus.TIMEOUT) or (exit_code is not None and exit_code != 0):
            err_msg = error_message or stdout_summary or f"Tool '{tool_name}' returned error code {exit_code}"
            culprit_file = f_write[0] if f_write else (f_read[0] if f_read else None)
            self.record_error(error_type=f"{tool_name}_Error", message=err_msg, file_path=culprit_file)
        elif status_enum == ToolExecutionStatus.SUCCESS and self.active_error:
            # If the tool succeeded on the file associated with the active error, clear it
            if self.active_error.get("file") and self.active_error["file"] in [os.path.normpath(x) for x in f_write]:
                self.clear_error()

        record = ToolRunRecord(
            call_id=f"call_{int(time.time() * 1000)}_{step_num}",
            step_number=step_num,
            tool_name=tool_name,
            arguments=dict(arguments),
            status=status_enum,
            task_id=tid,
            subtask_id=sub_id,
            intent=intent,
            exit_code=exit_code,
            stdout_summary=stdout_summary,
            raw_output_ref=raw_output_ref,
            files_read=f_read,
            files_written=f_write,
            files_deleted=f_del,
            duration_ms=duration_ms,
            error_message=error_message,
        )

        self.tool_history.append(record)
        if self.active_task:
            self.active_task.tool_history.append(record)
            self.active_task.updated_at = time.time()

        self.recalculate_progress()
        self._refresh_working_entities()


        return record

    def get_recent_tool_runs(self, limit: int = 5) -> List[ToolRunRecord]:
        """Returns the most recent N tool run records."""
        return self.tool_history[-limit:]

    # ------------------------------------------------------------------------
    # Background Job Bus
    # ------------------------------------------------------------------------

    def register_background_task(
        self,
        job_id: str,
        command: str,
        cwd: str = "",
        pid: Optional[int] = None,
        summary: str = "",
        workspace: str = "",
    ) -> BackgroundTaskRecord:
        """Registers a new asynchronous background subprocess."""
        effective_cwd = cwd or workspace or self.active_workspace_path or os.getcwd()
        tid = self.active_task.task_id if self.active_task else ""
        record = BackgroundTaskRecord(
            job_id=job_id,
            task_id=tid,
            command=command,
            cwd=effective_cwd,
            pid=pid,
            status="RUNNING",
            summary=summary or f"Running: {command[:60]}",
        )
        self.background_tasks[job_id] = record

        if self.active_task and job_id not in self.active_task.active_background_jobs:
            self.active_task.active_background_jobs.append(job_id)
            self.active_task.updated_at = time.time()

        self._refresh_working_entities()
        return record

    def update_background_task(
        self,
        job_id: str,
        status: Optional[str] = None,
        exit_code: Optional[int] = None,
        log_tail: Optional[str] = None,
        summary: Optional[str] = None,
    ) -> Optional[BackgroundTaskRecord]:
        """Updates real-time status, exit code, and log tail for a background job."""
        record = self.background_tasks.get(job_id)
        if not record:
            return None

        if status:
            record.status = status
            if status in ("COMPLETED", "FAILED", "CANCELLED"):
                record.end_time = time.time()
                if self.active_task and job_id in self.active_task.active_background_jobs:
                    self.active_task.active_background_jobs.remove(job_id)

        if exit_code is not None:
            record.exit_code = exit_code
        if log_tail is not None:
            record.log_tail = log_tail
        if summary is not None:
            record.summary = summary

        if self.active_task:
            self.active_task.updated_at = time.time()

        self._refresh_working_entities()
        return record

    def get_background_tasks(self, active_only: bool = False) -> List[BackgroundTaskRecord]:
        """Lists registered background tasks."""
        if active_only:
            return [b for b in self.background_tasks.values() if b.status == "RUNNING"]
        return list(self.background_tasks.values())

    # ------------------------------------------------------------------------
    # Scratchpad Management
    # ------------------------------------------------------------------------

    def add_scratchpad_note(self, category: str, text: str) -> None:
        """Appends a note to a structured scratchpad category."""
        cat_lower = str(category).lower().strip()
        if "hypo" in cat_lower:
            self.scratchpad.add_hypothesis(text)
        elif "obs" in cat_lower:
            self.scratchpad.add_observation(text)
        elif "road" in cat_lower or "warn" in cat_lower or "block" in cat_lower:
            self.scratchpad.add_roadblock(text)
        elif "next" in cat_lower or "act" in cat_lower or "plan" in cat_lower:
            self.scratchpad.add_next_action(text)
        else:
            self.scratchpad.add_observation(text)

        self._refresh_working_entities()

    def clear_scratchpad_category(self, category: str) -> None:
        """Clears items from a specific scratchpad compartment."""
        cat_lower = str(category).lower().strip()
        if "road" in cat_lower or "block" in cat_lower:
            self.scratchpad.clear_roadblocks()
        elif "next" in cat_lower or "act" in cat_lower:
            self.scratchpad.clear_next_actions()

    # ------------------------------------------------------------------------
    # State Registers (Files, Variables, Errors)
    # ------------------------------------------------------------------------

    def record_file_touch(self, file_path: str) -> None:
        """Tracks files inspected, modified, or written during the session."""
        if not file_path:
            return
        norm_path = os.path.normpath(file_path)
        self.active_files.add(norm_path)
        self._refresh_working_entities()

    def record_error(self, error_type: str, message: str, file_path: Optional[str] = None) -> None:
        """Sets active execution exception / error state with un-truncated diagnostics."""
        self.active_error = {
            "type": error_type,
            "message": str(message).strip(),
            "file": os.path.normpath(file_path) if file_path else None,
            "timestamp": time.time(),
        }
        self.recalculate_progress()
        self._refresh_working_entities()

    def clear_error(self) -> None:
        """Clears active error upon successful resolution."""
        self.active_error = None
        self.recalculate_progress()
        self._refresh_working_entities()

    def set_variable(self, key: str, value: Any) -> None:
        """Registers a key-value session state variable."""
        self.active_variables[key] = value

    def set_terminal_register(self, key: str, output: Any) -> None:
        """Caches recent command or tool output."""
        self.terminal_registers[key] = output

    # ------------------------------------------------------------------------
    # Dynamic Attention Mask & Working Entity Computation
    # ------------------------------------------------------------------------

    def _refresh_working_entities(self) -> None:
        """
        Recomputes the active entity attention mask E_working:
        E_working = Tokens(root_goal)
                  U { basename(f) for f in active_files }
                  U subtasks[current_subtask_id].entities
                  U { active_error["type"], basename(active_error["file"]) }
                  U scratchpad tokens
        """
        mask: Set[str] = set()

        # 1. Goal tokens
        if self.root_goal:
            tokens = re.findall(r"\b[A-Za-z0-9_-]{3,}\b", self.root_goal)
            for t in tokens:
                mask.add(t)

        # 2. File basenames and extensions
        for f in self.active_files:
            base = os.path.basename(f)
            if base:
                mask.add(base)
                name_without_ext = os.path.splitext(base)[0]
                if name_without_ext:
                    mask.add(name_without_ext)

        # 3. Active subtask entities
        if self.current_subtask_id and self.current_subtask_id in self.subtasks:
            task = self.subtasks[self.current_subtask_id]
            for ent in task.entities:
                mask.add(ent)

        # 4. Active error tokens
        if self.active_error:
            err_type = self.active_error.get("type")
            if err_type:
                mask.add(err_type)
            err_file = self.active_error.get("file")
            if err_file:
                base = os.path.basename(err_file)
                mask.add(base)
                name_no_ext = os.path.splitext(base)[0]
                if name_no_ext:
                    mask.add(name_no_ext)

        # 5. Scratchpad tokens (roadblocks & observations)
        for obs in self.scratchpad.observations[-3:]:
            for tok in re.findall(r"\b[A-Za-z0-9_-]{4,}\b", obs):
                mask.add(tok)

        self.working_entities = mask

    def get_working_attention_seeds(self) -> Set[str]:
        """Returns the current volatile attention seeds for retrieval biasing."""
        self._refresh_working_entities()
        return set(self.working_entities)

    # ------------------------------------------------------------------------
    # Slotted Quota Working Context Formatting
    # ------------------------------------------------------------------------

    def format_working_context_block(self, max_tokens: int = 400) -> str:
        """
        Generates a high-density, budgeted Markdown context block for prompt injection.
        Guarantees complete error messages and key tool history without exceeding limits.
        """
        lines: List[str] = ["### [ACTIVE TASK & WORKING STATE]"]

        # 1. Goal and Lifecycle Phase
        phase_str = self.active_task.phase.value if self.active_task else "IDLE"
        goal_str = self.root_goal or (self.active_task.goal if self.active_task else "None")
        lines.append(f"- **Goal**: {goal_str} (Phase: {phase_str})")

        # 2. Workspace Root
        ws = self.get_workspace_path()
        if ws:
            status_tag = "[Pinned]" if self.workspace_cache_scope == "task" else f"[TTL: {self.workspace_cache_ttl} turns]"
            lines.append(f"- **Workspace**: `{ws}` {status_tag}")

        # 3. Progress & Active Milestone
        prog = self.recalculate_progress()
        lines.append(
            f"- **Progress**: [{prog.completed_subtasks}/{prog.total_subtasks} subtasks complete] ({prog.percent_complete}%)"
        )
        if prog.current_milestone:
            lines.append(f"- **Milestone**: {prog.current_milestone}")

        # 4. Background Tasks
        active_bgs = self.get_background_tasks(active_only=True)
        if active_bgs:
            bg_strs = [f"[{b.job_id}] (PID {b.pid or '?'}) {b.command[:35]} (Status: {b.status})" for b in active_bgs[:2]]
            lines.append(f"- **Background Jobs**: {', '.join(bg_strs)}")

        # 5. Active Files
        if self.active_files:
            file_names = [os.path.basename(f) for f in list(self.active_files)[-4:]]
            lines.append(f"- **Active Files**: {', '.join(file_names)}")

        # 6. Active Execution Error (Uncut full message)
        if self.active_error:
            err_file = f" in {os.path.basename(self.active_error['file'])}" if self.active_error.get("file") else ""
            lines.append(f"- **ACTIVE ERROR**{err_file}: `{self.active_error.get('type')}` - {self.active_error.get('message')}")

        # 7. Structured Scratchpad (Hypotheses, Observations, Roadblocks, Next Steps)
        sp = self.scratchpad
        has_scratchpad = sp.hypotheses or sp.observations or sp.roadblocks or sp.next_actions
        if has_scratchpad:
            lines.append("\n### [SCRATCHPAD & RECENT FINDINGS]")
            if sp.hypotheses:
                lines.append(f"- **Working Theory**: {sp.hypotheses[-1]}")
            if sp.observations:
                obs_list = [f"'{o}'" for o in sp.observations[-3:]]
                lines.append(f"- **Observations**: {'; '.join(obs_list)}")
            if sp.roadblocks:
                lines.append(f"- **Roadblock**: {sp.roadblocks[-1]}")
            if sp.next_actions:
                lines.append(f"- **Next Action**: {sp.next_actions[-1]}")

        # 8. Recent Tool Runs Telemetry
        recent_tools = self.get_recent_tool_runs(limit=3)
        if recent_tools:
            lines.append("\n### [RECENT TOOL EXECUTION TELEMETRY]")
            for tr in recent_tools:
                dur = f"{int(tr.duration_ms)}ms" if tr.duration_ms > 0 else "<5ms"
                exit_str = f"Exit: {tr.exit_code}" if tr.exit_code is not None else "No Exit Code"
                status_icon = "SUCCESS" if tr.status == ToolExecutionStatus.SUCCESS else tr.status.value
                sum_text = tr.stdout_summary[:140] if tr.stdout_summary else (tr.error_message[:140] if tr.error_message else "Executed")
                lines.append(f"- `{tr.tool_name}` [{status_icon}] ({exit_str}, {dur}) -> {sum_text}")

        return "\n".join(lines)


# Alias for explicit enterprise semantic consumers
ExecutiveWorkingMemoryEngine = WorkingMemoryEngine
