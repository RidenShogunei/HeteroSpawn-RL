# Validation report index

Files in this directory are immutable, redacted experiment evidence. They record what was tested
at a particular commit and are not current setup instructions. Use the
[end-to-end guide](../runbooks/qwen3-wideseek-end-to-end.md) to run the project.

## Current Qwen3/WideSeek evidence

- [Held-out SFT/RL comparison](2026-07-27-qwen3-4b-heldout-sft-rl-comparison.md) — current
  checkpoint-only comparison and its no-improvement conclusion.
- [Bounded SFT-to-RL pilot](2026-07-26-qwen3-4b-bounded-sft-rl-pilot.md) — shared and
  independent one-cycle training path.
- [Multi-step SFT](2026-07-26-qwen3-4b-wideseek-multistep-sft.md) — 192-task warm start.
- [1,024-token output profile](2026-07-26-qwen3-4b-wideseek-1024-output.md) — reason for the
  current generation allowance.
- [Qwen3 QLoRA WideSeek environment](2026-07-24-qwen3-4b-qlora-wideseek.md) — full model and
  environment contract.
- [Offline retrieval acceptance](2026-07-23-wideseek-remote-full-acceptance.md) — pinned
  Wiki-2018/Qdrant/E5 deployment.

## Contract evidence

The LocalHF, exact-token episode cycle, WideSeek agent loop, data/reward, offline retrieval, and
phase-transaction reports document lower-level acceptance. They remain useful when changing a
contract, but they do not supersede the current Qwen3 guide.

## Historical and optional experiments

Reports for MiniMax, xbench, vLLM, RLinf, verl, and OpenRLHF preserve earlier capability findings.
Those integrations remain diagnostic or held-out paths. In particular, the native RLinf, verl,
and OpenRLHF spikes were blocked on the tested Turing host and are not the active training
backend.
