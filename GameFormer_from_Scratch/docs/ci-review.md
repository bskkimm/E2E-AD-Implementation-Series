# CI, review and server enforcement

The actual series-root workflow is `../.github/workflows/gameformer.yml`; its
reviewable installation source is [ci/gameformer.yml](../ci/gameformer.yml).
Docs checks require them to match. Every PR and push to main runs all checks,
without path filters. Eleven isolated matrix jobs cover Python 3.11 and 3.12
unit/integration checks plus independent pre-commit, lint/format, types, notebook,
package, dependencies and docs/config checks. An always-running **GameFormer
required checks** aggregate rejects failed, cancelled and skipped requirements.

Actions are SHA-pinned, permissions read-only, checkout credentials unpersisted,
job timeouts set, superseded runs cancelled, and failure logs retained for 14 days.
Tests use `pull_request`, never secret-bearing `pull_request_target`. No persistent
GPU runner or private playbook access is needed. Local stage commands and logs are
described in [the command guide](../COMMAND_GUIDE.md).

The local pre-commit hook checks whitespace/final newlines, conflict markers,
private keys, YAML/TOML/JSON validity, file size and empty notebook outputs. The
notebook stays below 1 MiB; only the generated platform lock has a 2 MiB allowance.
Ruff covers Python and notebook cells; pinned actionlint checks workflow/shell
syntax. The scientific unit job enforces at least 90% line coverage across
geometry, data, model and losses. Other interfaces have dedicated behavioral
tests; optional SDK/GPU branches are not claimed as tested. Subprocess CLI/notebook
execution is tested directly and is not counted by parent-process coverage.

Delivery status must be assessed separately from checked-in policy:

- SSH authenticates as `bskkimm` and can publish signed topic commits.
- The connected GitHub app reads repository/PR/check metadata. Branch protection
  reads returned **403 Resource not accessible by integration**; rulesets read as
  empty. Existing classic protection therefore cannot be verified or configured
  using that connection. No claim of branch-enforced checks or review is made.
- The existing `pra` service is active. Its live own-repository policy was read
  from `bskkimm/pr-agent-automation/config/cloud.toml`; no private-provider or
  detect-only setting was changed. `gh auth status` reports the stored token
  invalid, so the CLI cannot dispatch its trusted review workflow or configure
  protection. A schedule alone does not guarantee review of this PR's head.
- The fresh whole-branch scientific/engineering reviewer inspected `fd62eff` and
  found five Important issues. All were accepted and fixed at `35ca03d`, with
  observed failing-then-passing regressions and 32 passing tests. The
  [review report](whole-branch-review.md) records every finding and ruling. This
  local reviewer is distinct from the installed trusted pra integration, and an
  AI COMMENT is distinct from a human approval.

All nine local stages passed against source `35ca03d` and refreshed evidence:
scientific-core coverage 93.98%; exact-lock audit queried 165 dependencies with
zero known advisories and no skipped packages. The original hosted run passed
ten matrix jobs but failed host-dependent notebook asset regeneration. The fix
uses wheel-bundled fonts and Pillow; the current hosted outcome and head SHA are
recorded in [draft PR #1](https://github.com/bskkimm/E2E-AD-Implementation-Series/pull/1).
This historical local record is separate from that live PR status.

Until actual hosted checks and trusted current-head review are observed, delivery remains
an open draft with unmet external gates. A missing reviewer or empty comment list
does not mean a pass. Restoring valid `gh` authentication is required to request
`pra cloud run --pr <PR URL> --force` and inspect/configure classic protection.
For feedback on this project's own PR, use `pra fix <PR URL>` after reviewing the
feedback-loop skill. Do not merge, auto-approve, force-push or resolve threads.
