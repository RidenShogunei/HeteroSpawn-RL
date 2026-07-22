# Security Policy

## Credentials

- Never commit API keys, tokens, `.env` files, decrypted benchmark answers, or authorization headers.
- Local credentials are provided through environment variables or an OS credential manager.
- GitHub Actions credentials use repository/environment secrets and least-privilege permissions.
- If a credential is pasted into an issue, pull request, log, commit, or chat, revoke it immediately and rotate it before continuing.

## Reporting

Until a dedicated security contact is established, report vulnerabilities privately to the repository owner through GitHub's private vulnerability reporting feature. Do not open a public issue containing secrets or exploit details.
