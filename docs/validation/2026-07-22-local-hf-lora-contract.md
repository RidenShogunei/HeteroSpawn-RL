# LocalHF LoRA single-GPU contract validation

Date: 2026-07-22

## Configuration

- GPU: NVIDIA GeForce RTX 4060 Laptop GPU, 8 GB
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Model revision: `7ae557604adf67be50417f59c2c2f167def9a775`
- Model weight SHA-256: `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`
- Precision: FP16
- LoRA: rank 8, alpha 16, dropout 0, `q_proj` and `v_proj`
- Runtime: PyTorch 2.4.0+cu121, Transformers 4.57.6, PEFT 0.19.1, NumPy 1.26.4

The model was loaded from an existing local directory only after verifying that its weight digest matched the pinned upstream blob. No API credential or benchmark plaintext was used.

## Result

The opt-in smoke passed every contract check:

- actual generated token IDs aligned one-to-one with generation log-probabilities;
- Main update changed only the Main train adapter while both rollout and Sub adapters stayed unchanged;
- Main sync published `MR0 → MR1`, after which `MR0` generation was rejected;
- four Sub requests shared `SR0` and the same serialized single-GPU endpoint;
- Sub update changed only the Sub train adapter, and sync published `SR0 → SR1`;
- checkpoint adapter, optimizer, and RNG state passed digest validation and restore;
- no generated response was decoded and re-encoded for training.

The final validation run allocated 1,344,967,168 peak GPU bytes and completed in 12.76 seconds after the verified model was cached. The machine-readable report and binary checkpoints remain under ignored `artifacts/local-contract/`.

This is a reference-contract validation, not a verl/RLinf selection, benchmark result, or production training claim.
