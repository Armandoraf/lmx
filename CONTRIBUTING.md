# Contributing to LMX

Thanks for helping improve LMX. Please discuss substantial API or provider
changes in an issue before opening a pull request.

## Local setup

Required tools are Rust 1.89+, Python 3.11+, Node.js 18+, and npm. Install
Python packaging tooling with `python -m pip install maturin`.

```sh
cargo test --workspace
cd python && maturin develop && python -m unittest discover -s tests
cd ../typescript && npm ci && npm test
```

Provider behavior belongs in `crates/lmx-core`. Bindings should only expose the
core’s behavior idiomatically; do not add separate provider, streaming, or
image-processing logic in Python or TypeScript.

## Tests

Every behavior change needs a core test or cross-language fixture and an
adapter-level test where the public API changes. Tests must not require live
provider credentials.

## Security and credentials

Do not commit credentials, access tokens, or provider responses containing
sensitive data. LMX supports documented provider authentication only; do not
add code that reads, refreshes, or repurposes third-party application OAuth
credentials.

Report vulnerabilities privately as described in [SECURITY.md](SECURITY.md).

## Releases

All packages share the version in the workspace. Before publishing a release:

1. Update the workspace, Python, and TypeScript package versions together.
2. Update [CHANGELOG.md](CHANGELOG.md).
3. Run the complete test matrix and clean-install checks.
4. Create and push a version tag after the release commit reaches `main`.
5. Publish Python to PyPI and TypeScript/native packages to npm using trusted
   publishing or the release workflow.

Do not publish a package from an untagged or unverified commit.
