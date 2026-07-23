# WideSeek remote full-environment acceptance

Date: 2026-07-23

## Scope

This acceptance deployed the complete pinned Wiki-2018/Qdrant/E5 environment on RTX 2080 Ti
hardware, exercised one deterministic Search-to-Access agent episode against the real services,
and ran short shared-policy and independent Main/Sub LoRA cycles with the pinned real
`Qwen/Qwen2.5-0.5B-Instruct` model.

MiniMax was used only as the non-official development semantic Judge. These results validate the
environment, rollout, reward, update, synchronization, transaction, and recovery contracts; they
are not comparable to official WideSeek results and do not claim reward improvement.

## Fixed identities and assets

- WideSeek/RLinf upstream:
  `d9f3d8a9db4d7aad1d641029293295503dd3eb2c`
- Training data:
  `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Wiki-2018 Corpus:
  `178d7d037f661be3159b0c3a8a4119b974f01880`
- Model:
  `Qwen/Qwen2.5-0.5B-Instruct@7ae557604adf67be50417f59c2c2f167def9a775`
- Corpus manifest:
  3,400 files, 155,895,995,164 bytes,
  `7f22b16e05f90d2fd7d0ff724effc1eb7cc543d5207526125e26d458cb8a4aa5`
- E5 manifest:
  nine files, 438,900,149 bytes,
  `5877db5cb6f70f910ee862f852515faedb391a15f4558ddb2ed3f2b86f8c88be`
- Environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`

The official Hub endpoint timed out repeatedly, so the standard Hugging Face client switched to
the configured mirror. Every downloaded byte was then checked against the same source-independent
trusted manifest. The complete corpus passed verification before deployment; partial download
parts were removed only after their final files had been rehashed.

## Offline service result

The pinned Qdrant service reported the collection green with 26,134,257 points and indexed
vectors, cosine vectors of size 768, and HNSW `m=32` / `ef_construct=512`. The access service
loaded 5,903,530 pages. E5 ran on one RTX 2080 Ti while Qdrant and page access used host
CPU/storage.

The deterministic environment smoke produced one Main spawn, one Sub, five model steps, and the
ordered tool sequence `search -> access`. Exact token/log-probability alignment, stable event
indices, and Search provenance on Access all passed. The policy actions in this smoke are
scripted intentionally; the Search and Access calls use the complete real offline environment.

## Real-model training results

Both runs used two complete system rollouts per task, a 4,096-token sequence limit, a 512-token
generation limit, and one RTX 2080 Ti for LocalHF generation and LoRA optimization.

| Check | Shared / `width_20k:0` | Independent / `depth_20k:1` |
| --- | ---: | ---: |
| Status | passed | passed |
| Phase shape | joint update | Main update, fresh Sub phase |
| Complete rollouts | 2 | 2 per phase |
| Exact token/log-prob round-trip | passed | passed |
| Atomic phase commits | 1 | 2 |
| Development-Judge provider calls | 11 | 0 |
| Peak allocated VRAM | 8,948,772,864 bytes | 2,394,334,208 bytes |
| Elapsed time | 146.61 s | 6.33 s |
| Spawned Sub instances | 0 | 0 |
| Degenerate reward groups | 1 | 1 per phase |
| Optimizer transitions | shared 0 -> 1 | Main 0 -> 1; Sub remains 0 |
| Adapter parameter change | no | no |

The greedy 0.5B model answered directly in these sampled tasks. Their rollout rewards were equal,
so normalized advantages, gradients, and adapter changes were zero. The independent Sub phase
therefore exercised the specified empty-batch commit without advancing its optimizer or rollout
revision. Nonzero Main/Sub adapter mutation and partner isolation remain covered by the opt-in
LocalHF GPU contract fixture, which passed all six tests on the same machine.

This is an important boundary of the result: the complete system is operational, but the base
0.5B model has not learned to spawn reliably. SFT or learned-policy training data is still needed
before a real-model run can be expected to produce non-empty Sub updates consistently.

## Replacement-process recovery

A new backend process restored each committed phase:

| Topology | Recovered phases | Result |
| --- | ---: | --- |
| Shared | 1 | weight/optimizer identity preserved; new deployment identity; adapters synchronized |
| Independent | 2 | Main step 1 and empty Sub step 0 both recovered with the same guarantees |

The recovery reports contain one append-only recovery manifest for shared and two for independent.
No logical optimizer step was repeated and no stale deployment revision was reused.

## Findings fixed during acceptance

- The offline launcher previously ran its final readiness command from the Qdrant directory, so
  relative manifest defaults failed after all services had loaded. It now supplies absolute
  project manifest paths and has a regression test.
- Two concurrent identical semantic-Judge requests could both miss the cache and race to publish
  different provider-response digests. Per-cache-key single-flight now performs one provider
  request, and the waiter reuses the resulting cache entry.
- Terminating the original launcher stopped its direct retrieval parent but left spawned E5
  workers on the GPU. Qdrant and retrieval now start in dedicated sessions, and cleanup signals
  the complete process groups with a bounded TERM-to-KILL fallback. An isolated parent/child
  process-group probe passed, and the two verified orphan workers from the acceptance attempts
  were stopped; all nine GPUs returned to one MiB reported use.

## Runtime and regression checks

- Python 3.11.14
- PyTorch 2.4.0+cu121
- Transformers 4.47.1
- PEFT 0.14.0
- Accelerate 1.2.1
- Safetensors 0.5.2
- Default suite: 116 passed, one opt-in GPU module skipped, on both development and remote hosts
- Remote LocalHF GPU contract: 6 passed
- Ruff, format check, and mypy: passed

## Evidence handling

Ignored safe-report SHA-256 values:

- Offline rollout:
  `c69ee3ca43803d2c02b6ca25490018da50ab92549d362db290a405524edb967e`
- Shared training:
  `f4836b5b8f120d29960a908691a320cdc79a5e4d3a27dff94a8e0bee5097f758`
- Shared recovery:
  `a908a8a4111affa4ce92f790fb32dde7d6e3ffc77cda86ffc140085f781c4ab3`
- Independent training:
  `29e0463c2345783f00a4f415b959fcde98eaab0b105cbe4baaba554f6921b9c2`
- Independent recovery:
  `9f45e966ac52bb5afb635851f087a8e427b40d7787329d504a36b1c436c0a701`

Only revision identities, counts, timings, resource measurements, version transitions, and
digests are committed. Runtime reports, checkpoints, token arrays, model text, retrieved content,
reference answers, Judge responses, network identities, and credentials remain ignored.
