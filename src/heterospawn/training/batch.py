"""Derive exact-token training batches without text reconstruction."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from math import sqrt
from statistics import fmean

from heterospawn.domain.ids import EpisodeId, PolicyId, StepId
from heterospawn.domain.training import (
    PolicyTrainingBatch,
    PolicyTrainingSample,
    TrainingPhase,
    TrajectoryStep,
    canonical_digest,
    training_batch_digest_payload,
)
from heterospawn.domain.versions import WeightVersion


@dataclass(frozen=True)
class OutcomeAdvantageGroup:
    """Normalized system-rollout rewards, including episodes without role samples."""

    advantages: tuple[tuple[EpisodeId, float], ...]
    reward_mean: float
    reward_std: float
    degenerate: bool

    def as_mapping(self) -> dict[EpisodeId, float]:
        return dict(self.advantages)


def normalize_outcome_advantages(
    episode_rewards: Mapping[EpisodeId, float],
    *,
    epsilon: float = 1e-8,
) -> OutcomeAdvantageGroup:
    """Normalize one task/phase system-rollout group with population std."""

    if len(episode_rewards) < 2:
        raise ValueError("an outcome advantage group requires at least two rollouts")
    ordered = tuple(sorted(episode_rewards.items(), key=lambda item: str(item[0])))
    rewards = tuple(reward for _, reward in ordered)
    mean = fmean(rewards)
    std = sqrt(fmean((reward - mean) ** 2 for reward in rewards))
    degenerate = std == 0.0
    advantages = tuple(
        (episode_id, 0.0 if degenerate else (reward - mean) / (std + epsilon))
        for episode_id, reward in ordered
    )
    return OutcomeAdvantageGroup(
        advantages=advantages,
        reward_mean=mean,
        reward_std=std,
        degenerate=degenerate,
    )


class TrainingBatchBuilder:
    """Copies rollout values and derives masks, advantages, and balance weights."""

    def build(
        self,
        *,
        batch_id: str,
        phase: TrainingPhase,
        target_policy_id: PolicyId,
        expected_base_version: WeightVersion,
        steps: tuple[TrajectoryStep, ...],
        episode_advantages: Mapping[EpisodeId, float],
        loss_masks: Mapping[StepId, tuple[int, ...]] | None = None,
    ) -> PolicyTrainingBatch:
        selected = tuple(step for step in steps if step.policy_id == target_policy_id)
        agent_counts = Counter((step.episode_id, step.agent_instance_id) for step in selected)
        episode_agents: dict[EpisodeId, set[object]] = {}
        for step in selected:
            episode_agents.setdefault(step.episode_id, set()).add(step.agent_instance_id)

        samples: list[PolicyTrainingSample] = []
        for step in selected:
            if step.episode_id not in episode_advantages:
                raise ValueError(f"missing advantage for episode {step.episode_id}")
            mask = (
                loss_masks[step.step_id]
                if loss_masks is not None and step.step_id in loss_masks
                else (1,) * len(step.response_ids)
            )
            step_count = agent_counts[(step.episode_id, step.agent_instance_id)]
            agent_count = len(episode_agents[step.episode_id])
            samples.append(
                PolicyTrainingSample(
                    task_id=step.task_id,
                    episode_id=step.episode_id,
                    rollout_id=step.rollout_id,
                    source_step_id=step.step_id,
                    agent_role=step.agent_role,
                    agent_instance_id=step.agent_instance_id,
                    policy_id=step.policy_id,
                    rollout_revision=step.rollout_revision,
                    prompt_ids=step.prompt_ids,
                    response_ids=step.response_ids,
                    old_log_probs=step.response_log_probs,
                    loss_mask=mask,
                    advantage=episode_advantages[step.episode_id],
                    aggregation_weight=1.0 / (step_count * agent_count),
                )
            )

        sample_tuple = tuple(samples)
        payload = training_batch_digest_payload(
            batch_id=batch_id,
            phase=phase,
            target_policy_id=target_policy_id,
            expected_base_version=expected_base_version,
            samples=sample_tuple,
        )
        return PolicyTrainingBatch(
            batch_id=batch_id,
            phase=phase,
            target_policy_id=target_policy_id,
            expected_base_version=expected_base_version,
            samples=sample_tuple,
            batch_digest=canonical_digest(payload),
        )

    def build_from_rewards(
        self,
        *,
        batch_id: str,
        phase: TrainingPhase,
        target_policy_id: PolicyId,
        expected_base_version: WeightVersion,
        steps: tuple[TrajectoryStep, ...],
        episode_rewards: Mapping[EpisodeId, float],
        loss_masks: Mapping[StepId, tuple[int, ...]] | None = None,
    ) -> tuple[PolicyTrainingBatch, OutcomeAdvantageGroup]:
        group = normalize_outcome_advantages(episode_rewards)
        return (
            self.build(
                batch_id=batch_id,
                phase=phase,
                target_policy_id=target_policy_id,
                expected_base_version=expected_base_version,
                steps=steps,
                episode_advantages=group.as_mapping(),
                loss_masks=loss_masks,
            ),
            group,
        )
