# PilotNet Index Update Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Mark PilotNet as implemented in the series index and link its completed public repository.

**Architecture:** Make a documentation-only change in `README.md`. Keep implementation details in the dedicated PilotNet repository and update only the series-level status, link, and concise scope note.

**Tech Stack:** Markdown, Git.

## Global Constraints

- Keep the index concise.
- Do not claim an exact reproduction of NVIDIA's private training data or reported driving performance.
- Do not change the status or descriptions of models 2 through 9.

---

### Task 1: Update PilotNet Index Documentation

**Files:**
- Modify: `README.md:3,47,57-69`

**Interfaces:**
- Consumes: Public repository URL `https://github.com/bskkimm/PilotNet-from-scratch`.
- Produces: Accurate PilotNet status and repository link in the series index.

- [ ] **Step 1: Update the overview sentence**

Change the statement that every model is planned to state that PilotNet is implemented and the remaining models are planned.

- [ ] **Step 2: Update the PilotNet table row**

Replace `🔜 Planned` with `✅ Implemented` and replace `(coming soon)` with a Markdown link to `https://github.com/bskkimm/PilotNet-from-scratch`.

- [ ] **Step 3: Update the PilotNet model note**

Link the implementation repository and describe the implemented scope as the PilotNet architecture, CSV behavioral-cloning workflow, validation entry point, public dataset documentation, and documented limitations around NVIDIA's private data and reported performance.

- [ ] **Step 4: Update the remaining-status sentence**

Replace `All entries are plans, not completed reproductions.` with wording that identifies PilotNet as implemented and models 2 through 9 as planned.

- [ ] **Step 5: Verify the documentation diff**

Run:

```bash
git diff --check
git diff -- README.md
```

Expected: no whitespace errors; only PilotNet status, link, overview, and note wording change.

- [ ] **Step 6: Verify stale PilotNet placeholders are gone**

Run:

```bash
rg -n "PilotNet|coming soon|All entries are plans|Implemented|Planned" README.md
```

Expected: PilotNet has the repository link and implemented status; planned wording remains only for the other model entries.

- [ ] **Step 7: Commit the README update**

```bash
git add README.md
git commit -m "Mark PilotNet implementation in series index"
```
