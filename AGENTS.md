# Agent instructions

- Read the architecture baseline and relevant ADRs before changing behavior.
- Keep domain and orchestration modules independent of MiniMax, xbench, verl, RLinf, and search providers.
- Use structured role, policy, episode, and revision fields; never infer them from prompts.
- Never print, persist, or commit credentials. Live API tests must be opt-in and skipped when credentials are absent.
- Do not store decrypted xbench prompts or answers in tracked files, snapshots, or test output.
- Run `python -m pytest`, `python -m ruff check .`, `python -m ruff format --check .`, and `python -m mypy src` before handoff.
- Any change to fresh-rollout semantics, policy sharing, loss aggregation, primary rewards, backend choice, or benchmark fairness requires an ADR.
- Before remote GPU or backend-spike work, read `docs/runbooks/remote-backend-spike.md`, run the credential-safe preflight, pin every external revision, and keep framework objects out of core modules.
