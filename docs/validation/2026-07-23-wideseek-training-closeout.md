# WideSeek training closeout validation

Date: 2026-07-23

## Scope

This validation covers the HeteroSpawn-owned WideSeek rollout-to-training path. It uses the pinned
real `Qwen/Qwen2.5-0.5B-Instruct` model with real FP16 LoRA generation and optimization on one
RTX 4060 Laptop GPU. Search and Access used an in-memory WideSeek-shaped HTTP transport, so these
runs are explicitly `controlled-fixture`, not proof that the complete 156 GB corpus was deployed.

The complete corpus/Qdrant/E5 acceptance remains a separate remote run. MiniMax was not called,
and these results are not comparable to official WideSeek results.

## Fixed identities

- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Training-data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Upstream environment revision: `d9f3d8a9db4d7aad1d641029293295503dd3eb2c`
- Data split/task: `hybrid_20k`, index 0
- Complete system rollouts per task/phase: 2
- Sequence/generation limits: 4096/32 tokens
- Runtime: PyTorch `2.4.0+cu121`, Transformers `4.47.1`, PEFT `0.14.0`,
  Accelerate `1.2.1`, Safetensors `0.5.2`

## Results

| Check | Shared policy | Independent Main/Sub |
| --- | ---: | ---: |
| Status | passed | passed |
| Phase shape | joint update | Main update, fresh Sub phase |
| Exact token/log-prob round-trip | passed | passed |
| Atomic phase commits | 1/1 | 2/2 |
| Checkpoint restore | shared passed | Main passed; Sub had no checkpoint |
| Peak allocated VRAM | 3,407,363,584 bytes | 3,411,688,960 bytes |
| Cycle elapsed time | 4.14 s | 6.59 s |
| Spawned Sub instances | 0 | 0 in both phases |
| Degenerate reward groups | 1 | 1 per phase |
| Adapter parameter change | no | no |

The fixed greedy 0.5B policy answered directly in all four observed episodes. Both task groups
therefore had identical rewards and zero outcome advantage. The backend correctly performed a
zero-gradient shared/Main optimizer transaction, published the corresponding immutable
checkpoint and rollout revision, and skipped the empty independent Sub batch without advancing
its optimizer or rollout revision. This proves the closed-loop and empty-batch semantics, but it
does not prove learning progress or a non-empty WideSeek Sub update.

Independent nonzero Main/Sub LoRA mutation and isolation remain covered by the LocalHF contract
fixture, where explicit nonzero advantages are used. A remote acceptance task that must exercise
the learned spawning path should use `--require-sub-update`; a 0.5B base model may require SFT or
an appropriately learned policy before it reliably emits the WideSeek tool protocol.

## Additional findings

The first real run rejected every tool-bearing request because LocalHF validated only the base
chat-template revision. The fix records and accepts only revisions issued by its own encoder for
the exact tool schema; an unissued revision is still rejected.

Windows also translated the exported PEFT `adapter_config.json` to CRLF after its digest was
calculated over LF. Writing the canonical UTF-8 bytes directly made rollout-artifact identity
stable across Windows and Linux.

The recovery audit also found that the initial LocalHF deployment identifier was deterministic
from the device and policy name. It is now unique per backend process. An optional tiny-Qwen
contract test starts a replacement backend, restores the committed adapter, optimizer and RNG
checkpoint, synchronizes the recovered weights, and verifies both that the adapter hashes match
and that the replacement rejects the terminated process's rollout revision.

## Evidence handling

Ignored reports:

- Shared report SHA-256:
  `25900f60f2583534e87be7d06d8f5408ebc15d2ec031927172428677a536d161`
- Independent report SHA-256:
  `79394ac994b6052154841dffa61ec079c58f0ac92a7ab445cef83b675899e714`

The reports contain revisions, counts, digests, timings, resource measurements, and version
transitions only. They do not contain model text, reference answers, token arrays, retrieved
content, checkpoints, host addresses, or credentials.
