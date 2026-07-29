"""Grounded full-behavior WideSeek SFT construction (ADR-0010)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field

from heterospawn.benchmarks.wideseek import WideSeekDataset
from heterospawn.domain.ids import TaskId
from heterospawn.domain.training import canonical_digest
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
from heterospawn.search.base import AccessRequest, ResearchToolService, SearchRequest
from heterospawn.training.wideseek_sft import (
    SupervisedConversation,
    WideSeekRoleSftConstructor,
    WideSeekSftConstruction,
    WideSeekSftConstructionSummary,
)

WIDESEEK_GROUNDED_SFT_CONSTRUCTOR_SCHEMA = "heterospawn-wideseek-role-sft-v2"
_TOKEN_RE = re.compile(r"[A-Za-z0-9]{3,}")
_FULL_BEHAVIORS = (
    "main_spawn",
    "sub_search",
    "sub_access",
    "sub_summary",
    "main_final",
)


@dataclass(frozen=True, repr=False)
class _GroundedPartition:
    subtask: str
    query: str
    evidence_block: str
    search_message: str
    access_url: str
    access_content: str
    access_truncated: bool
    search_target: str
    access_target: str
    summary_target: str


class GroundingStats(BaseModel):
    """Answer-safe grounding counters for one construction pass."""

    model_config = ConfigDict(frozen=True, strict=True)

    attempted_tasks: int = Field(ge=0)
    grounded_tasks: int = Field(ge=0)
    ungrounded_tasks: int = Field(ge=0)

    @property
    def keep_rate(self) -> float:
        if self.attempted_tasks == 0:
            return 0.0
        return self.grounded_tasks / self.attempted_tasks


class WideSeekGroundedFullSftConstructor:
    """Build spawn/search/access/summary/final targets from real offline retrieval."""

    def __init__(
        self,
        *,
        max_workers: int = 4,
        search_topk: int = 5,
        max_search_message_results: int = 3,
        max_search_content_characters: int = 600,
        max_access_characters: int = 800,
        overlap_ratio: float = 0.4,
        min_token_hits: int = 2,
    ) -> None:
        if not 1 <= max_workers <= 4:
            raise ValueError("max_workers must be in 1..4")
        if not 1 <= search_topk <= 20:
            raise ValueError("search_topk must be in 1..20")
        if not 0.0 < overlap_ratio <= 1.0:
            raise ValueError("overlap_ratio must be in (0, 1]")
        if min_token_hits < 1:
            raise ValueError("min_token_hits must be >= 1")
        self._max_workers = max_workers
        self._search_topk = search_topk
        self._max_search_message_results = max_search_message_results
        self._max_search_content_characters = max_search_content_characters
        self._max_access_characters = max_access_characters
        self._overlap_ratio = overlap_ratio
        self._min_token_hits = min_token_hits
        self._partitioner = WideSeekRoleSftConstructor(max_workers=max_workers)
        self.constructor_revision = canonical_digest(
            {
                "schema": WIDESEEK_GROUNDED_SFT_CONSTRUCTOR_SCHEMA,
                "max_workers": max_workers,
                "search_topk": search_topk,
                "max_search_message_results": max_search_message_results,
                "max_search_content_characters": max_search_content_characters,
                "max_access_characters": max_access_characters,
                "overlap_ratio": overlap_ratio,
                "min_token_hits": min_token_hits,
                "prompt_revision": WIDESEEK_PROMPT_REVISION,
                "tool_schema_revision": WIDESEEK_TOOL_SCHEMA_REVISION,
                "behaviors": list(_FULL_BEHAVIORS),
                "excluded_targets": [],
                "grounding": "offline-wideseek-research-tool-service",
            }
        )

    async def build(
        self,
        dataset: WideSeekDataset,
        *,
        task_indices: tuple[int, ...],
        tools: ResearchToolService,
    ) -> WideSeekSftConstruction:
        if not task_indices:
            raise BenchmarkDataError("WideSeek grounded SFT task selection cannot be empty")
        if len(set(task_indices)) != len(task_indices):
            raise BenchmarkDataError("WideSeek grounded SFT task selection contains duplicates")
        tasks = dataset.tasks
        if any(index < 0 or index >= len(tasks) for index in task_indices):
            raise BenchmarkDataError("WideSeek grounded SFT task selection is out of range")

        conversations: list[SupervisedConversation] = []
        worker_counts: dict[int, int] = {}
        grounded = 0
        for task_index in task_indices:
            task = tasks[task_index]
            record = dataset.evaluator_record(task)
            reference = record.answers[0]
            blocks = self._partitioner._partition_reference(reference, record.is_markdown)
            partitions = await self._ground_partitions(
                task.task_id,
                task.prompt,
                blocks,
                tools=tools,
            )
            if partitions is None:
                continue
            grounded += 1
            worker_counts[len(partitions)] = worker_counts.get(len(partitions), 0) + 1
            conversations.extend(
                self._task_conversations(
                    task.task_id,
                    task.prompt,
                    dataset,
                    partitions,
                    self._partitioner._final_target(reference, record.is_markdown),
                )
            )

        if grounded < 1:
            raise BenchmarkDataError("no WideSeek tasks could be grounded for full-behavior SFT")

        behavior_counts = {
            behavior: sum(1 for item in conversations if item.behavior == behavior)
            for behavior in _FULL_BEHAVIORS
        }
        summary = WideSeekSftConstructionSummary(
            schema_revision=WIDESEEK_GROUNDED_SFT_CONSTRUCTOR_SCHEMA,
            dataset_revision=dataset.revision,
            source_digest=dataset.source_digest,
            split=dataset.split,
            constructor_revision=self.constructor_revision,
            selected_tasks=grounded,
            examples=len(conversations),
            main_final_examples=behavior_counts["main_final"],
            sub_summary_examples=behavior_counts["sub_summary"],
            main_spawn_examples=behavior_counts["main_spawn"],
            sub_search_examples=behavior_counts["sub_search"],
            sub_access_examples=behavior_counts["sub_access"],
            attempted_tasks=len(task_indices),
            grounded_tasks=grounded,
            ungrounded_tasks=len(task_indices) - grounded,
            worker_count_histogram=tuple(sorted(worker_counts.items())),
        )
        return WideSeekSftConstruction(tuple(conversations), summary)

    async def try_build_task(
        self,
        dataset: WideSeekDataset,
        *,
        task_index: int,
        tools: ResearchToolService,
    ) -> WideSeekSftConstruction | None:
        try:
            return await self.build(dataset, task_indices=(task_index,), tools=tools)
        except BenchmarkDataError:
            return None

    def _task_conversations(
        self,
        task_id: TaskId,
        question: str,
        dataset: WideSeekDataset,
        partitions: tuple[_GroundedPartition, ...],
        final_target: str,
    ) -> list[SupervisedConversation]:
        subtasks = tuple(item.subtask for item in partitions)
        spawn_target = "".join(
            WideSeekRoleSftConstructor._tool_call("subtask", {"subtask": subtask})
            for subtask in subtasks
        )
        main_system = Message(
            role="system",
            content=wideseek_main_system_prompt(
                max_main_rounds=3,
                max_spawn_per_round=4,
                max_spawn_per_episode=8,
            ),
        )
        main_user = Message(role="user", content=question)
        conversations: list[SupervisedConversation] = [
            self._conversation(
                task_id,
                dataset,
                "main",
                "main_spawn",
                (main_system, main_user),
                MAIN_TOOLS,
                spawn_target,
                partition_index=None,
            )
        ]

        worker_results = []
        for index, partition in enumerate(partitions):
            sub_system = Message(role="system", content=WIDESEEK_SUB_SYSTEM_PROMPT)
            sub_user = Message(role="user", content=partition.subtask)
            search_assistant = Message(role="assistant", content=partition.search_target)
            search_user = Message(role="user", content=partition.search_message)
            access_assistant = Message(role="assistant", content=partition.access_target)
            access_user = Message(
                role="user",
                content=WideSeekRoleSftConstructor._canonical_json(
                    [
                        {
                            "request_index": 0,
                            "tool": "access",
                            "status": "success",
                            "url": partition.access_url,
                            "content": partition.access_content,
                            "truncated": partition.access_truncated,
                        }
                    ]
                ),
            )
            conversations.append(
                self._conversation(
                    task_id,
                    dataset,
                    "sub",
                    "sub_search",
                    (sub_system, sub_user),
                    SUB_TOOLS,
                    partition.search_target,
                    partition_index=index,
                )
            )
            conversations.append(
                self._conversation(
                    task_id,
                    dataset,
                    "sub",
                    "sub_access",
                    (sub_system, sub_user, search_assistant, search_user),
                    SUB_TOOLS,
                    partition.access_target,
                    partition_index=index,
                )
            )
            conversations.append(
                self._conversation(
                    task_id,
                    dataset,
                    "sub",
                    "sub_summary",
                    (
                        sub_system,
                        sub_user,
                        search_assistant,
                        search_user,
                        access_assistant,
                        access_user,
                    ),
                    SUB_TOOLS,
                    partition.summary_target,
                    partition_index=index,
                )
            )
            worker_results.append(
                {
                    "agent_instance_id": f"sub-sft-{index}",
                    "subtask": partition.subtask,
                    "status": "success",
                    "content": partition.access_content,
                    "error_code": None,
                }
            )

        conversations.append(
            self._conversation(
                task_id,
                dataset,
                "main",
                "main_final",
                (
                    main_system,
                    main_user,
                    Message(role="assistant", content=spawn_target),
                    Message(
                        role="user",
                        content=(
                            "Delegated worker results, in request order:\n"
                            + WideSeekRoleSftConstructor._canonical_json(worker_results)
                        ),
                    ),
                ),
                MAIN_TOOLS,
                final_target,
                partition_index=None,
            )
        )
        return conversations

    async def _ground_partitions(
        self,
        task_id: TaskId,
        question: str,
        blocks: tuple[str, ...],
        *,
        tools: ResearchToolService,
    ) -> tuple[_GroundedPartition, ...] | None:
        grounded: list[_GroundedPartition] = []
        for index, block in enumerate(blocks):
            query = self._query_from_block(question, block)
            if not query:
                return None
            subtask = f"Collect evidence for partition {index + 1}: {query}"
            if len(subtask) > 2000:
                subtask = subtask[:1997] + "..."
            search = await tools.search(
                SearchRequest(
                    request_id=f"sft-ground:{task_id}:p{index}:search",
                    query=query,
                    max_results=self._search_topk,
                )
            )
            if not search.results:
                return None
            search_message = WideSeekRoleSftConstructor._canonical_json(
                [
                    {
                        "request_index": 0,
                        "tool": "search",
                        "status": "success",
                        "results": self._search_message_results(search.results),
                    }
                ]
            )
            search_target = WideSeekRoleSftConstructor._tool_call(
                "search",
                {"query": query, "topk": self._search_topk},
            )
            chosen = None
            for item in search.results:
                access = await tools.access(
                    AccessRequest(
                        request_id=f"sft-ground:{task_id}:p{index}:access",
                        url=item.url,
                        info_to_extract=subtask[:1000],
                        max_characters=self._max_access_characters,
                    )
                )
                if self._supports(access.content, block):
                    chosen = (item, access)
                    break
            if chosen is None:
                return None
            _item, access = chosen
            access_target = WideSeekRoleSftConstructor._tool_call(
                "access",
                {
                    "url": access.url,
                    "info_to_extract": "Extract the evidence required by the subtask.",
                },
            )
            grounded.append(
                _GroundedPartition(
                    subtask=subtask,
                    query=query,
                    evidence_block=block,
                    search_message=search_message,
                    access_url=access.url,
                    access_content=access.content,
                    access_truncated=access.truncated,
                    search_target=search_target,
                    access_target=access_target,
                    summary_target=f"Evidence summary:\n{access.content}",
                )
            )
        return tuple(grounded)

    def _search_message_results(self, results: tuple[object, ...]) -> list[dict[str, object]]:
        visible = results[: self._max_search_message_results]
        if not visible:
            return []
        per_result_budget = max(
            1,
            self._max_search_content_characters // len(visible),
        )
        payload: list[dict[str, object]] = []
        for item in visible:
            content = item.content  # type: ignore[attr-defined]
            payload.append(
                {
                    "title": item.title,  # type: ignore[attr-defined]
                    "url": item.url,  # type: ignore[attr-defined]
                    "content": content[:per_result_budget],
                    "content_truncated": len(content) > per_result_budget,
                    "score": item.score,  # type: ignore[attr-defined]
                }
            )
        return payload

    def _conversation(
        self,
        task_id: TaskId,
        dataset: WideSeekDataset,
        role: str,
        behavior: str,
        messages: tuple[Message, ...],
        tools: tuple,
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
            agent_role=role,  # type: ignore[arg-type]
            behavior=behavior,  # type: ignore[arg-type]
            dataset_revision=dataset.revision,
            source_digest=dataset.source_digest,
            constructor_revision=self.constructor_revision,
            messages=messages,
            tools=tools,
            target=target,
        )

    def _supports(self, evidence: str, reference_block: str) -> bool:
        tokens = tuple(dict.fromkeys(_TOKEN_RE.findall(reference_block.lower())))
        if not tokens:
            return False
        text = evidence.lower()
        hits = sum(1 for token in tokens if token in text)
        if hits == len(tokens):
            return True
        if hits < self._min_token_hits:
            return False
        return (hits / len(tokens)) >= self._overlap_ratio

    @staticmethod
    def _query_from_block(question: str, block: str) -> str:
        cells = [
            cell.strip()
            for line in block.splitlines()
            if line.strip().startswith("|") and line.strip().endswith("|")
            for cell in line.strip().strip("|").split("|")
            if cell.strip() and set(cell.strip()) != {"-"}
        ]
        if cells:
            seed = " ".join(cells[:8])
        else:
            seed = block.replace("Verified answer candidate:", "").strip()
        if not seed:
            seed = question.strip()
        query = " ".join(seed.split())
        return query[:400]
