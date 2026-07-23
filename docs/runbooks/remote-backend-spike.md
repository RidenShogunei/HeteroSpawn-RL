# Remote backend capability spike runbook

This runbook is the entry point for an agent working on the remote GPU host. Read `AGENTS.md`,
the architecture baseline's Milestone 2.5 section, applicable backend ADRs, and this file before
installing a framework or changing code.

## Scope and safety

- Framework spikes evaluate native training capabilities. A standalone rollout-engine spike has
  narrower acceptance criteria and must not be reported as a training-backend pass.
- Do not request, copy, print, or store API credentials. The spike uses synthetic prompts and local models.
- Keep framework-specific objects and configuration outside the domain and orchestration packages.
- Do not kill GPU processes, delete another checkout, use privileged containers, or occupy every GPU.
- Start with one explicitly selected idle GPU. Expand to two only when a native framework example requires it.
- Stop and report a blocker instead of applying unreviewed patches to verl/RLinf or HeteroSpawn core.

## Known host baseline

The credential-safe audit on 2026-07-23 observed Ubuntu 22.04, nine RTX 2080 Ti GPUs
with 11,264 MiB each, NVIDIA driver 580.82.07, Docker access, host CUDA toolkit 11.5,
Python 3.10.12, `uv`, and about 1.1 TB free disk. Treat this as historical evidence:
run preflight again before every spike.

The repository is public, so clone and fetch over HTTPS require no GitHub credentials. Push still
requires authentication supplied out of band. A Git bundle remains the fallback when the host
cannot reach GitHub reliably. Never place a token in a remote URL, shell history, repository file,
or report.

## Start every remote task

```bash
cd "$HOME/HeteroSpawn-RL"
git status --short --branch
git rev-parse HEAD
python3 scripts/remote_preflight.py \
  --output artifacts/backend-spikes/remote-preflight.json \
  --require-gpu-count 9
nvidia-smi
```

If the commit differs from the commit named in the task, stop. If any GPU is occupied, select
other idle indices with `CUDA_VISIBLE_DEVICES`; never terminate an existing process.

Runtime environments, model caches, framework clones, and downloaded containers belong under
`$HOME/heterospawn-runtime/`, not in the Git checkout. Reports and checkpoints belong
under ignored `artifacts/backend-spikes/<backend>/`.

## Candidate setup rules

1. Run every candidate in a separate container or virtual environment. Never install a candidate
   into the system Python or another candidate's environment.
2. Prefer an official container compatible with the installed NVIDIA driver. Do not build against
   the host CUDA 11.5 toolkit merely because it is on `PATH`.
3. Pin the upstream Git commit, image digest, Python, CUDA, PyTorch, rollout engine, and model
   revision before executing the native example. Do not use an unrecorded `latest` image.
4. Use `Qwen/Qwen2.5-0.5B-Instruct` at the revision recorded by HeteroSpawn's LocalHF validation
   unless the framework cannot load it. Record any substitution as a contract limitation.
5. First run the candidate's unmodified native generation/update/checkpoint example. Add a thin
   spike adapter only after the native path succeeds.

## Required experiments

Run the following in order for each candidate:

1. **Native startup**: initialize one training policy and one rollout endpoint on an explicitly
   selected GPU; record peak memory and startup time.
2. **Exact trajectory**: generate from token IDs and retain selected response IDs, per-token old
   log-probabilities, masks, stop reason, tokenizer revision, and sampling parameters without
   decode/re-encode.
3. **One update and sync**: perform one optimizer step, save the native checkpoint, synchronize
   rollout workers through the framework's native mechanism, and prove that stale rollout state
   is not reused.
4. **Checkpoint resume**: restart the native runner from the checkpoint and verify policy step,
   optimizer state, parameter digest, and generated rollout weight identity.
5. **Two-policy feasibility**: determine whether two independently versioned policies and
   optimizers can coexist, and whether 0, 1, and 4 runtime Sub instances can share one SubPolicy.
   This is a feasibility probe, not yet a production adapter.

Do not claim a pass from log text alone. Store machine-readable counts, revisions, hashes, timing,
and memory in `artifacts/backend-spikes/<backend>/report.json`.

## Stop conditions

Stop the candidate and record a structured blocker when any of these occurs:

- the pinned native example cannot run on Turing/RTX 2080 Ti;
- exact selected-token log-probabilities are unavailable;
- independent policy updates require modifying HeteroSpawn core or a large upstream fork;
- checkpoint resume cannot preserve optimizer/version identity;
- installation requires replacing host drivers or changing system packages;
- the smallest valid configuration still exceeds available memory.

Throughput cannot compensate for a correctness failure.

## Standalone vLLM Turing probe

ADR-0002 permits a standalone vLLM compatibility island for rollout only. On the known sm_75 host,
use the exact pins recorded in
`docs/validation/2026-07-23-vllm-turing-rollout-spike.md`; do not install an unbounded current
vLLM stack and do not reuse a framework environment.

Required rollout evidence is native startup without FA2, token-ID input and round-trip, one
selected-token log-probability per generated token, 0/1/4 request handling, LoRA loading, distinct
adapter digests across worker restarts, and resource measurements. Store machine-readable output
under `artifacts/backend-spikes/vllm-standalone/`. Any thin probe scripts belong in the isolated
runtime and are not product code or tracked repository fixtures.

This probe does not validate optimizer updates, checkpoint recovery, online adapter hot swap, or
stale-revision rejection. Those remain responsibilities of the project-owned training backend and
the future `PolicyService` adapter.

The product adapter authorized by ADR-0002 is now available under
`heterospawn.backends.vllm_rollout`. Its opt-in conformance command is
`heterospawn vllm-rollout-contract-smoke`; it composes LocalHF training with isolated vLLM
generation, uses restart synchronization, and writes a credential-safe report under ignored
`artifacts/`. Follow the README command and the product validation record rather than copying the
one-off spike scripts. Passing this command still does not turn vLLM into a training backend or
complete the deferred distributed-framework milestone.

## Handoff record

Commit only a credential-safe Markdown summary under `docs/validation/`. It must name the upstream
commit/image digest, dependency versions, selected GPU count, model revision, commands in abstracted
form, contract results, resource measurements, blockers, and report digest. Do not commit raw model
text, token arrays, checkpoints, framework caches, decrypted benchmark data, hostnames, usernames,
IP addresses, or credentials.

Suggested task prompt for the remote agent:

> Read AGENTS.md, the Milestone 2.5 architecture section, and the remote backend spike runbook.
> Run the credential-safe preflight first. Evaluate only the named backend and pinned revision using
> its native workflow. Do not change HeteroSpawn core contracts. Produce an ignored machine-readable
> report and a redacted validation summary; stop on any listed correctness or environment blocker.
