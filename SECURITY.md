# Security Policy

## Supported versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |

## Reporting a vulnerability

Please **do not open a public issue** for security or privacy problems.

Use GitHub's private vulnerability reporting:

<https://github.com/M0-AR/bladder/security/advisories/new>

Include:

- A description of the issue and its impact
- Steps to reproduce
- Affected version or commit
- Any suggested mitigation
- Whether patient data or a live deployment is involved

You can expect an acknowledgement within seven days. Please allow time for a fix and a coordinated disclosure.

## What counts as a security issue

- A path traversal, arbitrary file write, or command injection in the CLI
- Unsafe model or archive loading that executes untrusted code
- A dependency vulnerability with a realistic exploit path
- Accidental exposure of credentials, patient data, or private annotations
- A website issue that could mislead users about clinical validity or regulatory status

## What does not count

- Model accuracy on a dataset you cannot share
- A missing feature
- A disagreement with the research disclaimer
- Vulnerabilities in unsupported versions

## Data handling

This repository must never contain patient data. If you believe a file contains identifiable health information, report it privately and do not redistribute it. The maintainers will remove the file and may invalidate affected history.

## Deployment guidance

If you deploy this software in a research setting:

- Run it inside your institution's approved environment.
- Keep patient data on managed storage with access controls and audit logs.
- Encrypt data in transit and at rest.
- Log access to videos and derived frames.
- Define a retention and deletion policy.
- Complete the privacy impact assessment required by your jurisdiction.
- Do not expose an unauthenticated inference endpoint to the public internet.

This project is not a medical device and does not provide clinical decision support. See `DISCLAIMER.md`.