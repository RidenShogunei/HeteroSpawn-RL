# Public repository release audit

Date: 2026-07-23

## Purpose

Verify that changing HeteroSpawn-RL from private to public would not expose credentials, SSH
configuration, decrypted benchmark data, runtime artifacts, or model checkpoints.

## Audit

- Scanned the final `main` tree and all 11 known GitHub branch refs for API-token/private-key
  signatures and remote-host details.
- Scanned tracked paths for `.env`, private/decrypted data, artifacts, bundles, key files, and
  xbench CSV files. The only match was the intentional `.env.example`, whose credential values
  are empty.
- Scanned 21 GitHub Issues/PRs, one issue comment, and available review/commit comments.
- Confirmed there were no releases or GitHub Actions artifacts.
- Recomputed both ignored backend-spike report digests and verified that the tracked validation
  notes contained no raw model output, token arrays, hostname, username, IP address, or credential.

No secret or private-data finding blocked publication.

## Result

The GitHub repository visibility was changed to `PUBLIC`. Code history, commit metadata,
Issues/PRs, and Actions logs are therefore public; GitHub repository secrets remain inaccessible.
The remote GPU host is permitted to clone/fetch anonymously, but its first anonymous fetch test
failed with a transient TLS/network error. Credential-free Git bundles remain the documented
fallback, while push access still requires separate authentication.
