# WideSeek role-targeted SFT warm-start runbook

This runbook covers the optional pre-RL warm start selected by ADR-0006. The current implementation
constructs and audits examples; it does not yet perform a model update.

## Why construction is required

The pinned WideSeek training files contain questions, reference answers, and answer-format
metadata. They do not contain Main/Sub conversations, spawn actions, Search/Access histories, or
evidence summaries. A direct question-to-answer conversion is prohibited because it would train
Main to bypass delegation.

The deterministic constructor creates only:

- a `main_final` target after a legal spawn turn and ordered successful worker results;
- a `sub_summary` target after a legal Search-to-Access history.

Spawn count, subtask decomposition, query choice, source choice, and Access choice are not
supervised targets.

## Prerequisites

Prepare and verify the pinned training data:

```bash
heterospawn wideseek-fetch-assets \
  --manifest manifests/wideseek-train-data.json \
  --destination artifacts/wideseek-assets/train-data \
  --endpoint auto
```

The official endpoint is attempted first. The existing asset command may fall back to the
configured Hugging Face mirror only under its documented retry rules. Source identity always
comes from the committed manifest and content digests.

## Answer-safe dry run

```bash
heterospawn wideseek-sft-dry-run \
  --manifest manifests/wideseek-train-data.json \
  --data-dir artifacts/wideseek-assets/train-data \
  --split hybrid_20k \
  --task-limit 8 \
  --max-workers 4
```

Use repeated `--task-index` flags for an explicit deterministic selection. When they are present,
`--task-limit` is ignored.

The command prints only the pinned dataset/source/constructor revisions, aggregate role counts,
and the worker-count histogram. It does not print or persist questions, references, conversations,
token arrays, or per-example digests. Plaintext examples exist only in memory and are discarded
when the process exits.

## Contract boundary

`SupervisedTrainingBatch` is intentionally distinct from `PolicyTrainingBatch`. An SFT batch has
exact prompt and target token IDs plus a target-only loss mask; it has no rollout revision, old
log-probability, reward, advantage, or episode aggregation weight.

Do not:

- serialize constructed plaintext examples or add them to fixtures;
- build examples from xbench or any held-out evaluation task;
- label a dry run as successful training;
- route SFT examples through the RL batch builder;
- use the released WideSeek-R1-4B as a teacher without a new ADR.

The next implementation stage adds LocalHF QLoRA SFT update/checkpoint/sync/restore support, then
runs the unchanged fixed 16-task compliance profile as the declared readiness gate.
