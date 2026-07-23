# WideSeek multi-round agent-loop contract validation

- Date: 2026-07-23
- Upstream semantic reference:
  `RLinf@d9f3d8a9db4d7aad1d641029293295503dd3eb2c`
- Scope: CPU contract fixtures only; no model, dataset, Judge, or external service was used

## Implemented contract

- Main has at most three logical turns. A non-empty output without a tool call is `ANSWER`; one to
  four valid `subtask` calls create a spawn round. The episode total is capped at eight workers.
- Sub has at most four logical turns. It may issue up to three Search/Access calls per turn, and
  Access accepts only URLs returned by that same Sub's earlier Search.
- Tool calls in one turn execute concurrently, while trace insertion follows request order.
- Every episode owns a reserve/commit/release ledger. Tool and Sub failures become structured
  outcomes and do not cancel siblings.
- Every Main/Sub model call preserves policy/template/revision identity, prompt IDs, response IDs,
  selected-token old log-probabilities, sampling parameters, and stop reason.
- Canonical tool request/result JSON is retained with validating SHA-256 digests. Search and Access
  records include URL provenance and provider response identity.
- The one-round and WideSeek multi-round orchestrators implement one common rollout protocol, so
  the existing episode-balanced batch and fresh alternating coordinator can consume either.

## Automated evidence

The focused fixtures cover:

- 0-spawn direct answer;
- a four-worker spawn round followed by a one-worker spawn round;
- malformed Main output retained before a valid repair;
- Search then Access, forged URL rejection, and a failed Search alongside successful siblings;
- stable agent, tool, and event order despite intentionally different completion timing;
- a failed Sub policy call without sibling cancellation;
- episode spawn-budget overflow retained before repair;
- exact response-token and old-log-probability round-trip with no decode/re-encode in training
  records.

At merge time the complete default suite passed with the optional real-model test disabled by
default. Asset, evaluator, and real offline-retrieval validation belong to the following delivery
slices.
