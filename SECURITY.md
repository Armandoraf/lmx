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

LMX reads documented provider credentials from the environment. Its Codex
provider accepts only caller-supplied, request-scoped OAuth credentials and
sends them only to the Codex backend. It must never read another application's
credential files, persist credentials, refresh OAuth tokens, or substitute an
identity from a process-level cache.
