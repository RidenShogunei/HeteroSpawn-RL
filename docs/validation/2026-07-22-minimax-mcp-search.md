# MiniMax Token Plan MCP search validation — 2026-07-22

## Scope

- Official server package: `minimax-coding-plan-mcp==0.0.4`, pinned in the subprocess command.
- Transport: MCP stdio through the stable Python SDK v1 contract.
- Tool: `web_search`.
- Policy: real `MiniMax-M2.7` through the existing evaluation adapter.
- Output discipline: credentials, query result bodies, URLs, model answers, and reasoning were not persisted.

## Capability probe

- MCP initialization: passed.
- Advertised tools: `web_search`, `understand_image`.
- Real safe query provider status: `0`.
- Organic results: `10`.
- Related searches: `0`.
- Only counts and a SHA-256 response digest were printed.

## End-to-end conformance

The live conformance episode used MiniMax for both the Main/Sub policy and MiniMax Token Plan MCP for search:

- Main → Sub → Main terminal path: passed.
- Spawn count: `1`.
- Sub statuses: `success`.
- Stable events: `3`.
- Main attempts: `2`.
- Invalid Main attempts: `0`.
- Trace trainable: `false`, as required for an external API policy.

## Security and reproducibility decisions

- The MCP package version is pinned rather than using a floating `uvx` resolution.
- MCP stderr is discarded so provider diagnostics cannot enter ordinary logs.
- The child process receives an allowlist of necessary OS/network variables plus MiniMax configuration; unrelated API keys are not inherited.
- Provider result text is validated and reduced to immutable `SearchItem` values.
- The first implementation starts an isolated MCP process per query. A persistent session is a future throughput optimization, not required for contract validation.

## Limitations

- Search results are live and therefore nondeterministic; this backend is for API evaluation, not default RL training.
- No official xbench score is claimed. A benchmark batch and the pinned judge protocol remain separate work.
