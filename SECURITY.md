# Security policy

## Reporting a vulnerability

Please do not open a public issue for a suspected vulnerability. Email
armandorafael2057@gmail.com with a description, reproduction steps, impact,
and any suggested remediation. We will acknowledge reports within seven days
and provide a status update after triage.

## Supported versions

Security fixes are made on the latest released minor version. Users should
upgrade to the latest release before reporting an issue.

## Credential handling

LMX reads documented provider credentials from the environment. It must never
read, persist, refresh, or transmit ChatGPT/Codex OAuth credentials or another
application's tokens.
