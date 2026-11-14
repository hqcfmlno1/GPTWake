# Positive Processed Workflow Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement the tasks inline, with regression tests before behavior changes.

**Goal:** Give human-recording and synthetic teams a shared positive augmentation workflow that reads processed audio only.

**Architecture:** Keep the notebook as one standalone cell. Separate raw recording and preview from explicit saving of a reviewed phrase. A processed-file selector and explicit all-human and all-synthetic scopes share one augmentation action for recorded and externally prepared positives.

**Tech Stack:** Python, nbformat, ipywidgets, soundfile, audiomentations, unittest.

**Spec:** The user-supplied workflow: record indefinitely, Stop saves raw only; preview and manually select bounds; explicitly save processed; synthetic teams copy ready-made WAVs to `positive/synthetic/` and `negative/synthetic/`, treating them as processed; augment selected files, all human, or all synthetic separately. Remove the combined raw-to-six-variants action.

## Global Constraints

- Keep existing recordings and generated data intact.
- Keep exactly one standalone notebook code cell.
- Keep Markdown and UI documentation in English.
- Positive processed WAVs are nonempty mono 16 kHz phrases that fit Window; they are not padded until augmentation.
- External prepared synthetic WAVs do not require raw audio or JSON sidecars. Preserve supplied metadata when present and record an external processed source honestly when absent.
- Negative synthetic inputs are nonempty mono 16 kHz audio and retain their prepared duration. Hard-negative and noise workflows remain functional.
- Keep human processed and synthetic groups separate in all-files actions; selected-file actions may explicitly include either group.
- Same-name human and synthetic inputs must retain distinct lineage and augmentation runs.

## Review Focus

- A raw-only library must not enable or feed batch augmentation.
- An unsaved slider edit must not affect processed augmentation.
- Copying valid synthetic WAVs into synthetic must work without creating or inventing raw files.
- Oversized, stereo, or wrong-rate processed files must not stop other valid files from augmenting, and must produce visible explanations.
- Newly saved files are selected automatically; refreshing preserves valid selections and recording disables controls.

## Task 1: Separate preparation and augmentation

**Files:** notebook, tests/test_positive_workflow.py, tests/test_notebook_data_layout.py, tests/test_notebook_kernel_output.py.

**Interfaces:** `save_positive_sample()` saves the selected raw phrase only. `processed_positive_files()` discovers external and recorded processed WAVs. `refresh_positive_processed(selected=None)`, `update_positive_controls(change=None)`, and `augment_chosen_positives()` manage the selector and its selected/all-human/all-synthetic scopes. `augmentation_sources(kind, group=None)` discovers and filters the two input groups; hard negatives retain human inputs only.

- [x] Add regression tests for saving without augmentation, invalid bounds/window rejection, empty processed library, selected/all scopes, external files without JSON, preserved metadata, invalid processed formats, stale selections, and unsaved raw edits.
- [x] Observe the missing behavior fail before changing the notebook.
- [x] Replace the combined generation action with Save positive sample and processed-only augmentation controls.
- [x] Accept metadata-free positive processed files and validate audio before augmentation outputs are written.
- [x] Adapt existing preview tests to explicitly save before augmenting; retain the real-kernel duplicate-preview check.
- [x] Run the complete unittest suite.

## Task 2: Document and verify the team workflow

**Files:** README.md, data/README.md, the notebook's visible control descriptions.

- [x] Document human preparation and synthetic copy-in, format requirements, processed-only scopes, and per-file validation.
- [x] Verify lint, formatting, notebook validity, one-cell layout, cleared outputs, and diff whitespace.
- [x] Review changes for accidental raw reads during batch augmentation and unintended changes to negative categories.

## Verification and Review

- The original failing tests exposed missing explicit preparation and processed-only augmentation. Scope-isolation tests failed before synthetic folders were supported.
- Independent review found failed raw loads could leave the previous savable source active. This is fixed by clearing raw state before loading, with a regression test. Clearing the raw picker also disables Save and Preview.
- Oversized prepared positives are rejected from their header before audio decoding.
- The user refined the design during implementation: human uses processed/, synthetic uses synthetic/, and all-files actions select one group. No existing audio was moved or relabeled.
- Follow-up review of the revised group scopes found synthetic-negative augmentation could be blocked by unrelated malformed human metadata. The selected-group action now scans only its group, while combined Refresh reports and skips invalid human sidecars. Regression test observed failing before the fix.
- Final verification: 68 unittest cases passed; Ruff lint and formatting passed; notebook validation confirmed exactly one cleared standalone code cell.
