from pydantic import ValidationError

from heterospawn.domain.ids import PolicyId
from heterospawn.domain.versions import RoleBinding, RolloutRevision, WeightVersion


def test_rollout_revision_requires_matching_policy() -> None:
    weights = WeightVersion(
        policy_id=PolicyId("main_policy"),
        optimizer_step=0,
        checkpoint_digest="sha256:main-v0",
    )

    try:
        RolloutRevision(
            policy_id=PolicyId("sub_policy"),
            weight_version=weights,
            deployment_id="rollout-a",
            replica_set_revision=0,
        )
    except ValidationError:
        pass
    else:
        raise AssertionError("mismatched policy IDs must be rejected")


def test_shared_policy_is_explicit_in_role_bindings() -> None:
    bindings = (
        RoleBinding(role="main", policy_id=PolicyId("shared"), trainable=True),
        RoleBinding(role="sub", policy_id=PolicyId("shared"), trainable=True),
    )

    assert bindings[0].policy_id == bindings[1].policy_id
    assert {binding.role for binding in bindings} == {"main", "sub"}
