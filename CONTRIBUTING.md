# Contributing

## Workflow

1. Create or select a GitHub issue with explicit acceptance criteria.
2. Branch from `main` using `codex/<topic>` or `feature/<topic>`.
3. Add tests with the implementation and record architecture changes as ADRs.
4. Run `python -m pytest`, `python -m ruff check .`, `python -m ruff format --check .`, and `python -m mypy src`.
5. Open a pull request linked to the issue. Merge only after CI passes and the acceptance criteria are checked.

Commits should be small and intentional, using prefixes such as `docs:`, `test:`, `feat:`, `fix:`, or `chore:`.

## Project invariants

- Main and Sub are explicit roles; role inference from prompt text is forbidden.
- API policy output must never be represented as trainable rollout data when exact tokens or log-probabilities are unavailable.
- Benchmark ground truth is evaluator-only data and must not be exposed to policies.
- Provider-specific types do not cross adapter boundaries.
- Secrets and decrypted benchmark data never enter Git history or ordinary logs.
