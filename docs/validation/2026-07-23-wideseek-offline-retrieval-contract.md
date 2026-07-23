# WideSeek offline retrieval contract validation

- Date: 2026-07-23
- Upstream semantics:
  `RLinf@d9f3d8a9db4d7aad1d641029293295503dd3eb2c`
- Corpus:
  `RLinf/Wiki-2018-Corpus@178d7d037f661be3159b0c3a8a4119b974f01880`
- Retriever:
  `intfloat/e5-base-v2@f52bf8ec8c7124536f0efb74aca902b2995e5bcd`
- Environment revision:
  `2b90af41266aafdbf27af8743f7a80700567afaa24ec5993961b39150c193340`

## Trusted inventories

The corpus Hub inventory was resolved at the pinned revision with file metadata enabled:

- 3,400 files;
- 155,895,995,164 total bytes;
- 3,189 LFS objects bound by SHA-256;
- 211 regular Git objects bound by canonical Git blob SHA-1;
- manifest digest
  `7f22b16e05f90d2fd7d0ff724effc1eb7cc543d5207526125e26d458cb8a4aa5`.

The E5 runtime subset contains the safetensors weights, tokenizer, pooling, sentence-transformer,
and model configuration needed by the pinned server: nine files, 438,900,149 bytes, manifest
digest `5877db5cb6f70f910ee862f852515faedb391a15f4558ddb2ed3f2b86f8c88be`.

## Contract evidence

CPU fixtures exercise the exact upstream HTTP shapes:

- `/retrieve` sends one batch of `queries`, `topk`, and `return_scores=true`, then validates and
  maps `document.contents`, `document.url`, optional title, and score;
- `/access` sends a URL list, verifies aligned canonical URL provenance, applies the caller's
  character bound, and records truncation;
- retryable transport/HTTP failures use bounded backoff; schema and non-retryable failures are
  terminal structured tool failures;
- Qdrant readiness requires status `green`, vector size 768, cosine distance, HNSW `m=32`, and
  `ef_construct=512`;
- a Search-to-Access readiness probe returns only revision, counts, booleans, and collection
  configuration—never query, URL, snippet, or page text;
- non-LFS Git object verification and all existing SHA-256 asset behavior share one manifest
  contract.

The Linux launcher validates all immutable bytes before deployment, keeps the verified corpus
read-only, creates a mutable copy-on-write Qdrant deployment, pins the upstream checkout, and
stops both child services on exit.

This validation is a protocol/implementation result. The complete 156 GB service was not launched
on the development laptop. A full remote Search-to-Access run remains required before the PR5
training report can claim end-to-end real-environment acceptance.
