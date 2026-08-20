# PilotNet Index Update Design

## Goal

Update the series index so PilotNet reflects the existing implementation in
`bskkimm/PilotNet-from-scratch`, while keeping the index concise and avoiding
claims of reproducing NVIDIA's private dataset or published performance.

## Changes

- Change the PilotNet table status from planned to implemented.
- Replace the placeholder sub-repository text with a link to the PilotNet
  repository.
- Revise the repository overview so it says PilotNet is implemented and the
  remaining models are planned.
- Update the PilotNet model note to link the implementation and describe its
  current scope: architecture, CSV training workflow, validation, and public
  dataset documentation.
- Remove the obsolete `(coming soon)` wording for PilotNet only.

## Non-goals

- Do not copy source code, experiment artifacts, or the PilotNet README into
  this index repository.
- Do not change the status or descriptions of models 2 through 9.
- Do not claim an exact reproduction of NVIDIA's private training data or
  reported driving performance.

## Verification

- Confirm all PilotNet links point to the public repository.
- Search for stale PilotNet-only phrases such as `(coming soon)` and
  `🔜 Planned`.
- Review the final diff to ensure only the intended README and design document
  changed.
