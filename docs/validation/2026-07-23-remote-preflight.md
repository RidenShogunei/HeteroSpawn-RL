# RTX 2080 Ti remote host preflight

Date: 2026-07-23

## Purpose

Establish a credential-safe baseline before transferring the repository or installing verl/RLinf.
The SSH session performed read-only inspection only; no package, driver, container, or repository
was installed during this audit.

## Observed environment

- OS: Ubuntu 22.04, x86_64
- Kernel: 6.8.0-40-generic
- GPUs: 9 x NVIDIA GeForce RTX 2080 Ti, 11,264 MiB each
- NVIDIA driver: 580.82.07
- Driver-advertised CUDA compatibility: 13.0
- Host CUDA toolkit (`nvcc`): 11.5
- Python: 3.10.12
- Git: 2.34.1
- Docker: 28.5.1 with daemon access
- `uv`: available
- Home filesystem: approximately 1.1 TB free
- Existing HeteroSpawn checkout: none
- Private GitHub repository access: unavailable

## Consequences

- Transfer the repository without credentials using a Git bundle.
- Install Python 3.11 and all candidate dependencies in isolated runtime directories or containers.
- Prefer pinned official containers; do not treat the host CUDA 11.5 toolkit as the candidate build
  target merely because the newer driver can run newer CUDA containers.
- Select idle GPUs explicitly and start with one candidate at a time.
- Re-run `scripts/remote_preflight.py` before every experiment because this record is not live state.

This validates host visibility only. It does not establish verl/RLinf compatibility or complete
Milestone 2.5.
