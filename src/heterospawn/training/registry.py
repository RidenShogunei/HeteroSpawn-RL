"""Role-to-policy topology registry with explicit trainability."""

from __future__ import annotations

from heterospawn.domain.ids import PolicyId
from heterospawn.domain.training import TrainingPhase
from heterospawn.domain.versions import AgentRole, RoleBinding, RolloutRevision
from heterospawn.errors import ConfigurationError


class PolicyRegistry:
    def __init__(
        self,
        bindings: tuple[RoleBinding, ...],
        revisions: tuple[tuple[PolicyId, RolloutRevision], ...],
    ) -> None:
        binding_map: dict[AgentRole, RoleBinding] = {}
        for binding in bindings:
            if binding.role in binding_map:
                raise ConfigurationError(f"duplicate role binding: {binding.role}")
            binding_map[binding.role] = binding
        if "main" not in binding_map:
            raise ConfigurationError("main role binding is required")

        revision_map = dict(revisions)
        if len(revision_map) != len(revisions):
            raise ConfigurationError("duplicate policy revision")
        for binding in bindings:
            revision = revision_map.get(binding.policy_id)
            if revision is None:
                raise ConfigurationError(f"missing revision for policy {binding.policy_id}")
            if revision.policy_id != binding.policy_id:
                raise ConfigurationError("binding and revision policy mismatch")

        self._bindings = binding_map
        self._revisions = revision_map

    def binding(self, role: AgentRole) -> RoleBinding:
        try:
            return self._bindings[role]
        except KeyError:
            raise ConfigurationError(f"unbound role: {role}") from None

    def revision(self, policy_id: PolicyId) -> RolloutRevision:
        try:
            return self._revisions[policy_id]
        except KeyError:
            raise ConfigurationError(f"unknown policy: {policy_id}") from None

    def role_revision(self, role: AgentRole) -> RolloutRevision:
        return self.revision(self.binding(role).policy_id)

    def replace_revision(self, revision: RolloutRevision) -> None:
        if revision.policy_id not in self._revisions:
            raise ConfigurationError(f"unknown policy: {revision.policy_id}")
        self._revisions[revision.policy_id] = revision

    def target_for_phase(self, phase: TrainingPhase) -> PolicyId | None:
        if phase == "joint_update":
            main = self.binding("main")
            sub = self.binding("sub")
            if main.policy_id != sub.policy_id or not main.trainable or not sub.trainable:
                raise ConfigurationError("joint_update requires one shared trainable policy")
            return main.policy_id
        role: AgentRole = "main" if phase == "main_update" else "sub"
        binding = self._bindings.get(role)
        return binding.policy_id if binding is not None and binding.trainable else None

    @property
    def is_shared_trainable(self) -> bool:
        main = self._bindings.get("main")
        sub = self._bindings.get("sub")
        return bool(
            main is not None
            and sub is not None
            and main.policy_id == sub.policy_id
            and main.trainable
            and sub.trainable
        )

    def snapshot(self) -> tuple[tuple[PolicyId, RolloutRevision], ...]:
        return tuple(sorted(self._revisions.items(), key=lambda item: str(item[0])))
