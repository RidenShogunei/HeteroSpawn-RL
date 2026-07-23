# WideSeek data, mirror, evaluator, and reward validation

- Date: 2026-07-23
- Upstream environment: `RLinf@d9f3d8a9db4d7aad1d641029293295503dd3eb2c`
- Training-data revision: `47832ea20581f78d32cd6b32b4b37b985cbbc9df`
- Trusted manifest digest:
  `93d124ee9ea849fad828e542c7aff3ecac431be8e4f2ee17001d9b964b1ef49e`

## Real data verification

The official endpoint was unreachable from the validation host after three bounded connection
attempts. The configured Hugging Face mirror returned the pinned `X-Repo-Commit` and the three
files were verified against the committed trust manifest:

| Split | Files | Bytes | Records | Format |
| --- | ---: | ---: | ---: | --- |
| `width_20k` | 1 | 80,053,112 | 20,000 | 20,000 Markdown |
| `depth_20k` | 1 | 6,460,586 | 20,000 | 20,000 boxed |
| `hybrid_20k` | 1 | 43,464,582 | 20,000 | 9,999 Markdown / 10,001 boxed |

Total verified size was 129,978,280 bytes. Dataset inspection returned only safe aggregate fields;
it did not print questions or reference answers.

The mirror exposed a real compatibility edge: both locally available `huggingface_hub` 0.24.0 and
an isolated 1.24.0 rejected its redirected HEAD metadata despite successful standard resolve
downloads. The production downloader now tries `hf_hub_download` first and, only for that mirror
metadata failure, resumes through the same public pinned Hub resolve route. A fresh 6,460,586-byte
download through this path matched SHA-256
`b3fd543ddf81962ef140bc992f0eb2185c6abf3132ff0cc266f6532808b20b1a`.
Manifest verification remains the publication barrier regardless of transport.

## Automated evidence

The default CPU suite covers:

- three-attempt official-to-mirror fallback and terminal authorization/revision behavior;
- exact digest verification, corrupt-file quarantine, copied-resource verification, and safe paths;
- width, depth, and hybrid schema validation, duplicate IDs, revision checks, and answer isolation;
- strict Markdown and balanced boxed parsing, required columns, unique-key alignment, item F1, and
  optional semantic matching;
- revision-bound digest-only Judge cache, bounded format repair, and terminal provider failure;
- shared/Main/Sub reward totals, search and length components, and immutable reward revisions.

The full default test, lint, format, and type-check gates passed. No dataset file, reference answer,
Judge plaintext, generated text, credential, cache, or partial download was committed.
