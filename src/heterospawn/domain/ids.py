"""Strongly typed identifiers used across domain boundaries."""

from typing import NewType

TaskId = NewType("TaskId", str)
EpisodeId = NewType("EpisodeId", str)
RolloutId = NewType("RolloutId", str)
StepId = NewType("StepId", str)
AgentInstanceId = NewType("AgentInstanceId", str)
PolicyId = NewType("PolicyId", str)
CheckpointId = NewType("CheckpointId", str)
