# Security Policy

Lensemble is research software. It is not a production federation service, a
clinical system, or a safety-certified robotics stack.

## Supported versions

Security fixes are made on `main`. The event snapshot tagged
`v0.1.0-codex-hackathon-final` is preserved for reproducibility and does not
receive patches.

## Reporting a vulnerability

Use GitHub's private vulnerability-reporting form under the repository's
**Security** tab. Please do not disclose a suspected vulnerability in a public
issue, discussion, or pull request before it has been assessed.

Include:

- the affected commit and platform;
- a minimal reproducer or proof of concept;
- the expected and observed trust boundary;
- the likely impact; and
- any suggested remediation or disclosure constraints.

Relevant reports include raw-data or secret egress, artifact-integrity bypasses,
unsafe deserialization, dependency or workflow compromise, and failures in the
documented aggregation, privacy, or provenance boundaries.

Research-result disagreements, benchmark corrections, and non-security bugs
belong in the public issue tracker. Current limitations that are already
documented as non-claims are not vulnerabilities by themselves.

The maintainer will acknowledge a private report, reproduce it where possible,
coordinate a fix, and agree on disclosure timing with the reporter. No response
or remediation deadline is guaranteed for this volunteer research project.
