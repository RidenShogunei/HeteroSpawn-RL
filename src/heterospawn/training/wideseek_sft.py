"""Deterministic, role-targeted WideSeek supervised warm-start construction."""

from __future__ import annotations

import json
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.benchmarks.wideseek import WideSeekDataset, WideSeekSplit
from heterospawn.domain.ids import PolicyId, TaskId
from heterospawn.domain.supervised import (
    SupervisedBehavior,
    SupervisedTrainingBatch,
    SupervisedTrainingExample,
    supervised_batch_digest_payload,
    supervised_example_digest_payload,
)
from heterospawn.domain.training import canonical_digest
from heterospawn.domain.versions import AgentRole, WeightVersion
from heterospawn.errors import BenchmarkDataError
from heterospawn.orchestration.wideseek_actions import (
    MAIN_TOOLS,
    SUB_TOOLS,
    WIDESEEK_TOOL_SCHEMA_REVISION,
)
from heterospawn.orchestration.wideseek_episode import (
    WIDESEEK_PROMPT_REVISION,
    WIDESEEK_SUB_SYSTEM_PROMPT,
    wideseek_main_system_prompt,
)
from heterospawn.policies.base import Message
from heterospawn.policies.trainable import SupervisedPolicyCodec, ToolDefinition

WIDESEEK_SFT_CONSTRUCTOR_SCHEMA = "heterospawn-wideseek-role-sft-v1"


@dataclass(frozen=True, repr=False)
class SupervisedConversation:
    """Plaintext exists only in the private construction/materialization boundary."""

    example_id: str
    task_id: TaskId
    agent_role: AgentRole
    behavior: SupervisedBehavior
    dataset_revision: str
    source_digest: str
    constructor_revision: str
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...]
    target: str


class WideSeekSftConstructionSummary(BaseModel):
    """Answer-safe description of an in-memory supervised construction."""

    model_config = ConfigDict(frozen=True, strict=True)

    schema_revision: str = WIDESEEK_SFT_CONSTRUCTOR_SCHEMA
    dataset_revision: str = Field(min_length=1)
    source_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    split: WideSeekSplit
    constructor_revision: str = Field(pattern=r"^[0-9a-f]{64}$")
    selected_tasks: int = Field(ge=1)
    examples: int = Field(ge=2)
    main_final_examples: int = Field(ge=1)
    sub_summary_examples: int = Field(ge=1)
    worker_count_histogram: tuple[tuple[int, int], ...]
    plaintext_lifecycle: str = "memory-only until exact token materialization"


@dataclass(frozen=True, repr=False)
class WideSeekSftConstruction:
    conversations: tuple[SupervisedConversation, ...]
    summary: WideSeekSftConstructionSummary


class WideSeekRoleSftConstructor:
    """Build post-tool targets without supervising Main spawn or Sub tool choices."""

    def __init__(self, *, max_workers: int = 4) -> None:
        if not 1 <= max_workers <= 4:
            raise ValueError("max_workers must be in 1..4")
        self._max_workers = max_workers
        self.constructor_revision = canonical_digest(
            {
                "schema": WIDESEEK_SFT_CONSTRUCTOR_SCHEMA,
                "max_workers": max_workers,
                "prompt_revision": WIDESEEK_PROMPT_REVISION,
                "tool_schema_revision": WIDESEEK_TOOL_SCHEMA_REVISION,
                "behaviors": ["main_final", "sub_summary"],
                "excluded_targets": ["main_spawn", "sub_search", "sub_access"],
                "reference_partition": "markdown-table-balanced-rows|single-fact",
            }
        )

    def build(
        self,
        dataset: WideSeekDataset,
        *,
        task_indices: tuple[int, ...],
    ) -> WideSeekSftConstruction:
        if not task_indices:
            raise BenchmarkDataError("WideSeek SFT task selection cannot be empty")
        if len(set(task_indices)) != len(task_indices):
            raise BenchmarkDataError("WideSeek SFT task selection contains duplicate indices")
        tasks = dataset.tasks
        if any(index < 0 or index >= len(tasks) for index in task_indices):
            raise BenchmarkDataError("WideSeek SFT task selection is out of range")

        conversations: list[SupervisedConversation] = []
        worker_counts: dict[int, int] = {}
        for task_index in task_indices:
            task = tasks[task_index]
            record = dataset.evaluator_record(task)
            reference = record.answers[0]
            blocks = self._partition_reference(reference, record.is_markdown)
            worker_counts[len(blocks)] = worker_counts.get(len(blocks), 0) + 1
            subtasks = tuple(
                f"Collect evidence partition {index + 1} of {len(blocks)} for the requested answer."
                for index in range(len(blocks))
            )
            conversations.append(
                self._main_final_conversation(
                    task.task_id,
                    task.prompt,
                    dataset,
                    subtasks,
                    blocks,
                    self._final_target(reference, record.is_markdown),
                )
            )
            conversations.extend(
                self._sub_summary_conversation(
                    task.task_id,
                    dataset,
                    subtask,
                    block,
                    index,
                )
                for index, (subtask, block) in enumerate(zip(subtasks, blocks, strict=True))
            )

        main_count = len(task_indices)
        summary = WideSeekSftConstructionSummary(
            dataset_revision=dataset.revision,
            source_digest=dataset.source_digest,
            split=dataset.split,
            constructor_revision=self.constructor_revision,
            selected_tasks=main_count,
            examples=len(conversations),
            main_final_examples=main_count,
            sub_summary_examples=len(conversations) - main_count,
            worker_count_histogram=tuple(sorted(worker_counts.items())),
        )
        return WideSeekSftConstruction(tuple(conversations), summary)

    def _main_final_conversation(
        self,
        task_id: TaskId,
        question: str,
        dataset: WideSeekDataset,
        subtasks: tuple[str, ...],
        blocks: tuple[str, ...],
        target: str,
    ) -> SupervisedConversation:
        spawn_content = "".join(
            self._tool_call("subtask", {"subtask": subtask}) for subtask in subtasks
        )
        worker_results = [
            {
                "agent_instance_id": f"sub-sft-{index}",
                "subtask": subtask,
                "status": "success",
                "content": block,
                "error_code": None,
            }
            for index, (subtask, block) in enumerate(zip(subtasks, blocks, strict=True))
        ]
        messages = (
            Message(
                role="system",
                content=wideseek_main_system_prompt(
                    max_main_rounds=3,
                    max_spawn_per_round=4,
                    max_spawn_per_episode=8,
                ),
            ),
            Message(role="user", content=question),
            Message(role="assistant", content=spawn_content),
            Message(
                role="user",
                content=(
                    "Delegated worker results, in request order:\n"
                    + self._canonical_json(worker_results)
                ),
            ),
        )
        return self._conversation(
            task_id,
            dataset,
            "main",
            "main_final",
            messages,
            MAIN_TOOLS,
            target,
            partition_index=None,
        )

    def _sub_summary_conversation(
        self,
        task_id: TaskId,
        dataset: WideSeekDataset,
        subtask: str,
        evidence: str,
        partition_index: int,
    ) -> SupervisedConversation:
        source_key = canonical_digest(
            {
                "task_id": task_id,
                "partition_index": partition_index,
                "constructor_revision": self.constructor_revision,
            }
        )[:20]
        url = f"wideseek://synthetic/{source_key}"
        search_result = [
            {
                "request_index": 0,
                "tool": "search",
                "status": "success",
                "results": [
                    {
                        "title": "Pinned training evidence",
                        "url": url,
                        "content": "A verified source is available for this evidence partition.",
                    }
                ],
            }
        ]
        access_result = [
            {
                "request_index": 0,
                "tool": "access",
                "status": "success",
                "url": url,
                "content": evidence,
                "truncated": False,
            }
        ]
        messages = (
            Message(role="system", content=WIDESEEK_SUB_SYSTEM_PROMPT),
            Message(role="user", content=subtask),
            Message(
                role="assistant",
                content=self._tool_call("search", {"query": subtask, "topk": 5}),
            ),
            Message(role="user", content=self._canonical_json(search_result)),
            Message(
                role="assistant",
                content=self._tool_call(
                    "access",
                    {
                        "url": url,
                        "info_to_extract": "Extract the evidence required by the subtask.",
                    },
                ),
            ),
            Message(role="user", content=self._canonical_json(access_result)),
        )
        return self._conversation(
            task_id,
            dataset,
            "sub",
            "sub_summary",
            messages,
            SUB_TOOLS,
            f"Evidence summary:\n{evidence}",
            partition_index=partition_index,
        )

    def _conversation(
        self,
        task_id: TaskId,
        dataset: WideSeekDataset,
        role: AgentRole,
        behavior: SupervisedBehavior,
        messages: tuple[Message, ...],
        tools: tuple[ToolDefinition, ...],
        target: str,
        *,
        partition_index: int | None,
    ) -> SupervisedConversation:
        example_id = "wideseek-sft:" + canonical_digest(
            {
                "task_id": task_id,
                "role": role,
                "behavior": behavior,
                "partition_index": partition_index,
                "dataset_revision": dataset.revision,
                "source_digest": dataset.source_digest,
                "constructor_revision": self.constructor_revision,
            }
        )
        return SupervisedConversation(
            example_id=example_id,
            task_id=task_id,
            agent_role=role,
            behavior=behavior,
            dataset_revision=dataset.revision,
            source_digest=dataset.source_digest,
            constructor_revision=self.constructor_revision,
            messages=messages,
            tools=tools,
            target=target,
        )

    def _partition_reference(
        self,
        reference: str,
        is_markdown: bool,
    ) -> tuple[str, ...]:
        if not is_markdown:
            return (f"Verified answer candidate: {reference.strip()}",)
        table_lines = tuple(
            line.strip()
            for line in reference.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
        )
        if len(table_lines) < 3:
            return (reference.strip(),)
        header = table_lines[:2]
        rows = table_lines[2:]
        worker_count = min(self._max_workers, len(rows))
        chunks = self._balanced_chunks(rows, worker_count)
        return tuple("\n".join((*header, *chunk)) for chunk in chunks)

    @staticmethod
    def _balanced_chunks(rows: tuple[str, ...], count: int) -> tuple[tuple[str, ...], ...]:
        quotient, remainder = divmod(len(rows), count)
        chunks: list[tuple[str, ...]] = []
        cursor = 0
        for index in range(count):
            size = quotient + (1 if index < remainder else 0)
            chunks.append(rows[cursor : cursor + size])
            cursor += size
        return tuple(chunks)

    @staticmethod
    def _final_target(reference: str, is_markdown: bool) -> str:
        target = reference.strip()
        if is_markdown or "\\boxed{" in target:
            return target
        return f"\\boxed{{{target}}}"

    @staticmethod
    def _tool_call(name: str, arguments: dict[str, object]) -> str:
        return (
            "<tool_call>"
            + WideSeekRoleSftConstructor._canonical_json({"name": name, "arguments": arguments})
            + "</tool_call>"
        )

    @staticmethod
    def _canonical_json(value: object) -> str:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


def materialize_supervised_conversations(
    conversations: tuple[SupervisedConversation, ...],
    codec: SupervisedPolicyCodec,
) -> tuple[SupervisedTrainingExample, ...]:
    """Tokenize prompt and target once; no plaintext is retained in returned records."""

    examples: list[SupervisedTrainingExample] = []
    for conversation in conversations:
        encoding = codec.encode_supervised(
            conversation.messages,
            conversation.target,
            conversation.tools,
        )
        loss_mask = (0,) * len(encoding.prompt_ids) + (1,) * len(encoding.target_ids)
        digest = canonical_digest(
            supervised_example_digest_payload(
                example_id=conversation.example_id,
                task_id=conversation.task_id,
                agent_role=conversation.agent_role,
                behavior=conversation.behavior,
                dataset_revision=conversation.dataset_revision,
                source_digest=conversation.source_digest,
                constructor_revision=conversation.constructor_revision,
                encoding=encoding,
                loss_mask=loss_mask,
            )
        )
        examples.append(
            SupervisedTrainingExample(
                example_id=conversation.example_id,
                task_id=conversation.task_id,
                agent_role=conversation.agent_role,
                behavior=conversation.behavior,
                dataset_revision=conversation.dataset_revision,
                source_digest=conversation.source_digest,
                constructor_revision=conversation.constructor_revision,
                encoding=encoding,
                loss_mask=loss_mask,
                example_digest=digest,
            )
        )
    return tuple(examples)


def build_supervised_training_batch(
    *,
    batch_id: str,
    target_policy_id: PolicyId,
    expected_base_version: WeightVersion,
    examples: tuple[SupervisedTrainingExample, ...],
) -> SupervisedTrainingBatch:
    payload = supervised_batch_digest_payload(
        batch_id=batch_id,
        target_policy_id=target_policy_id,
        expected_base_version=expected_base_version,
        examples=examples,
    )
    return SupervisedTrainingBatch(
        batch_id=batch_id,
        target_policy_id=target_policy_id,
        expected_base_version=expected_base_version,
        examples=examples,
        batch_digest=canonical_digest(payload),
    )
