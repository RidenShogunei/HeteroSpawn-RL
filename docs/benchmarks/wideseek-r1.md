# WideSeek-R1 training environment

WideSeek-R1 is the primary reinforcement-learning environment for HeteroSpawn-RL. The environment
semantics are pinned to upstream `RLinf` commit
`d9f3d8a9db4d7aad1d641029293295503dd3eb2c`; the training data is pinned to revision
`47832ea20581f78d32cd6b32b4b37b985cbbc9df`.

The policy receives a provider-neutral `ResearchTask` containing the question, answer format, and
dataset revision. Reference answers, required columns, and unique-key columns remain inside the
evaluator-owned dataset record and never enter prompts, traces, reports, or training samples.

## Prepare and inspect training data

Install the optional standard Hugging Face client:

```bash
python -m pip install -e ".[wideseek]"
heterospawn wideseek-fetch-assets
```

`--endpoint auto` tries the official Hub three times for retryable transport or server failures,
then tries `https://hf-mirror.com`. `--endpoint official` and `--endpoint mirror` select one
source explicitly. `HF_ENDPOINT` may also select the official endpoint or the configured mirror.
Authentication failures, missing revisions, and integrity failures are terminal and do not trigger
blind source switching.

Every downloaded or machine-copied file is checked against
`manifests/wideseek-train-data.json`. The manifest binds the upstream repository and revision to
each file's exact byte size and SHA-256 digest. Interrupted transfers retain partial bytes for
resume. Corrupt completed files are moved under the ignored destination's `.quarantine` directory.

An offline machine can copy the three files into the destination and run:

```bash
heterospawn wideseek-fetch-assets --verify-only
```

Inspect each split without exposing reference answers:

```bash
heterospawn wideseek-inspect-data --split width_20k
heterospawn wideseek-inspect-data --split depth_20k
heterospawn wideseek-inspect-data --split hybrid_20k
```

The command reports only the pinned revision, source digest, task count, and answer-format counts.
All model, dataset, corpus, cache, trace, Judge, and checkpoint files stay in ignored runtime
directories.

## Evaluation and reward

Width examples use strict fenced Markdown tables. Depth examples use the last balanced
`\boxed{...}` answer. Hybrid selects its parser per record. Table evaluation validates required
columns, aligns rows by unique keys, removes duplicate keys, and computes item-level F1.

Exact normalized values are evaluated locally. Optional semantic equivalence uses the
provider-neutral Judge contract. The MiniMax implementation is a temperature-zero development
Judge with bounded concurrency, bounded format repair, and a revision-bound digest-only cache. A
terminal Judge failure fails the phase; it is never converted to reward zero. MiniMax results are
not official-comparable.

The reward record contains an auditable component breakdown and three totals:

- `shared`: outcome, format, successful Access credit, length penalty, and context penalty;
- `main`: shared total minus configured spawn, Search/Access, token, and invalid-action costs;
- `sub`: the full system outcome for MVP Sub credit.

Evaluator, Judge, cache, prompt, tool, dataset, corpus, and reward identities are phase-transaction
inputs. Recovery rejects environment drift rather than reusing stale rollouts.
