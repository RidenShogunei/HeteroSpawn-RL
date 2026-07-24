# Qwen3-4B QLoRA and WideSeek validation

- Date: 2026-07-24
- Status: Passed
- Officially comparable: No
- Training host: one NVIDIA GeForce RTX 2080 Ti for QLoRA; one separate RTX 2080 Ti for E5
- Source baseline: `b793f20e6ade3debe1c3a80f4e98a1313a0258b2`

## Pinned inputs

- Model: `Qwen/Qwen3-4B`
- Model revision: `1cfa9a7208912126459214e8b04321603b3df60c`
- Trusted model-manifest digest:
  `7d1f0f5002a45efc43d4800d460e1e11d9849dee17a587a34d003709c5e57ce9`
- Verified model inventory: 13 files and 8,060,926,626 bytes, including all three safetensors
  shards
- WideSeek data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Offline environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`

The official Hugging Face endpoint timed out during three bounded metadata attempts. The standard
Hugging Face client then used `https://hf-mirror.com`; every downloaded byte was checked against
the same source-independent manifest. No model cache, token, generated text, checkpoint, hostname,
or service address is committed.

## Runtime

- Python 3.11.14
- PyTorch 2.4.0+cu121
- Transformers 4.53.3
- PEFT 0.14.0
- Accelerate 1.2.1
- bitsandbytes 0.45.5
- Safetensors 0.5.2
- NumPy 1.26.4

The policy used NF4 4-bit base weights with double quantization, FP16 compute, gradient
checkpointing, rank-8 `q_proj`/`v_proj` LoRA, and separate train/rollout adapters. The bounded
WideSeek profile disabled thinking. Sampling used the raw policy distribution: temperature 1,
top-p 1, and top-k 0.

The retrieval service ran in a separate environment selected through
`HETEROSPAWN_RETRIEVAL_PYTHON`. The launcher reverified 3,400 corpus files and 155,895,995,164
bytes, plus all E5 files. Qdrant was green with 26,134,257 points and the real Search-to-Access
probe passed.

## Exact backend contract

The real-model `local-contract-smoke` passed in 48.764 seconds with 4,412,407,808 peak allocated
VRAM bytes. Main and Sub both produced non-zero gradients (`0.565669` and `0.536905`), changed only
their own train adapters, retained the old rollout until sync, rejected stale revisions, and
restored adapter, optimizer, and RNG state from immutable checkpoints. Four Sub requests shared
one Sub rollout revision. The ignored report SHA-256 was
`4f62bd0cca4127e30cec33d729fd8b25436c189203dd1a4fdd21c00a2c643060`.

The tiny-Qwen3 optional backend fixture also verified that raw categorical sampling records the
same token log-probability semantics recomputed during update.

## Real WideSeek cycle

The successful command used one `width_20k` task, `G=2`, shared policy, a 4096-token model limit,
at most 512 generated tokens per turn, at most three model-visible Search results, 600 displayed
Search-content characters, and 800 displayed Access characters. Complete response digests and URL
provenance remained in the ignored audit trace.

Results:

- elapsed time: 222.877 seconds
- peak allocated VRAM: 7,288,627,712 bytes
- episodes: 2
- zero-spawn episodes: 0
- total spawned Sub instances: 3
- real Search/Access tool outcomes: 10
- training samples: 18 (7 Main and 11 Sub)
- longest prompt: 1,774 tokens
- longest complete training sequence: 1,985 tokens
- normalized episode advantages: approximately `-1` and `+1`
- degenerate task groups: 0
- optimizer step: 0 -> 1
- loss: `0.0050023`
- gradient norm: `0.2590155`
- shared adapter changed: yes
- exact raw-trajectory token/log-probability round-trip: passed
- checkpoint restore, rollout sync, phase commit, and environment binding: passed

The committed phase-manifest digest was
`ff10026af993990e33de61297a191da937e10575646f664da908a317959e8be4`; the immutable checkpoint
digest was `e1c66710064d78b28b30d1e784e9e20acc42770b57eaef1a2f0b11965cb58ddf`. The ignored safe report
SHA-256 was `9ffcc6888c6e2e0dc0b785f780f6db040987ba73a92c7bf7590223f82e1df2ad`.

## Findings retained from failed attempts

The first thinking-enabled run completed the architecture cycle but both outputs reached the
1,024-token limit without a tool call. It produced a degenerate group and no adapter change. A
length-truncated direct answer is now invalid and retained as a repair attempt.

Non-thinking mode immediately produced legal spawning and real Search. Initial updates still ran
out of memory because the backend retained multiple computation graphs and because multi-turn
tool messages accumulated to 3,500-4,096 tokens. The final implementation:

1. performs mathematically equivalent per-sequence backward accumulation while preserving
   agent-instance, episode, and batch weighting;
2. versions deterministic model-visible Search/Access budgets;
3. keeps complete provider identity, response digests, and provenance outside the bounded display
   payload.

The successful run demonstrates a functioning rollout/reward/update/sync/recovery loop and an
initial non-zero policy update. It does not demonstrate reward improvement, convergence, or an
official WideSeek score. A longer experiment needs warm-start/SFT data, multiple task groups, a
held-out evaluation protocol, and a separately budgeted semantic Judge.
