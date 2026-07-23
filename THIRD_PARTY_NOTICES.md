# Third-party notices

## xbench-evals

The xbench-DeepSearch adapter interoperates with the encrypted dataset format and deterministic answer-extraction behavior from [xbench-ai/xbench-evals](https://github.com/xbench-ai/xbench-evals), pinned to commit `17c562192cc7e62215bfb98b65e9f8806fb95504`.

xbench-evals is distributed under the MIT License. Its repository and license remain the authoritative source; no decrypted benchmark data is redistributed by HeteroSpawn-RL.

## RLinf / WideSeek-R1

The WideSeek environment adapter interoperates with the multi-turn planner/worker, Search/Access,
and answer-format semantics published in
[RLinf WideSeek-R1](https://github.com/RLinf/RLinf/tree/d9f3d8a9db4d7aad1d641029293295503dd3eb2c/examples/agent/wideseek),
pinned to commit `d9f3d8a9db4d7aad1d641029293295503dd3eb2c`.

RLinf is distributed under the Apache License 2.0. HeteroSpawn-RL implements provider-neutral
contracts and does not redistribute RLinf runtime code, model weights, datasets, or the Wiki
corpus.

The optional offline environment interoperates with
[`RLinf/Wiki-2018-Corpus`](https://huggingface.co/datasets/RLinf/Wiki-2018-Corpus), pinned to
revision `178d7d037f661be3159b0c3a8a4119b974f01880`, and
[`intfloat/e5-base-v2`](https://huggingface.co/intfloat/e5-base-v2), pinned to revision
`f52bf8ec8c7124536f0efb74aca902b2995e5bcd`. Their upstream repositories contain the
authoritative licenses and notices. Only content-identity manifests are committed here; corpus,
index, executable, and model bytes are not redistributed.
