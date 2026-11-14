# Negative Categories and Window Extraction Plan

**Goal:** Separate hard negatives from ordinary negatives and extract windows from explicitly selected ranges of long recordings without loading the full file.

**Spec:** Approved discussion: separate `negative` and `hard_negative`, each with raw/processed/augmented; both label 0. Hard negatives are recorded individually. Ordinary negatives can stream for an hour, then users select time intervals and sample random or sliding windows. Keep source lineage and group splits by source.

**Architecture:** Extend the existing streaming recorder with a hard-negative category. Negative Stop saves raw and offers bounded preview only. A separate panel selects category/source, editable time ranges, sampling mode/count/stride, and source split. Extraction seeks/reads one window at a time. Hard-negative clip mode preserves a selected event, padding short clips to the target window and rejecting oversized clips. Store each extraction in a unique numbered processed folder and augment only existing processed windows. Retain original three cells plus an equivalent standalone cell.

**Tech Stack:** Python, NumPy, SoundFile, SciPy, ipywidgets, audiomentations, unittest.

## Constraints and Review Focus

- Preserve raw recordings, existing outputs, current user changes, and the prior preview fix.
- No countdown, heuristic boundaries, real microphone tests, automatic splitting of long recordings at Stop, or commit/push.
- All Markdown stays in English.
- Validate intervals and parameters before creating output files; no windows may cross approved range boundaries.
- Random positions are sampled without replacement; overlapping windows are allowed and retain the same source group. Sliding windows omit partial tails.
- Default split is unassigned. An extraction locks its source split; later extraction cannot assign another split. This is source-level consistency, not automatic speaker/session partitioning.
- Reads are bounded to a window (or at most five preview seconds), including native-rate stereo inputs.
- Hard-negative padding never cuts away the selected event. Both negative categories retain label 0 through augmentation.

## Steps

- [x] Add failing tests for category storage, bounded random/sliding extraction, padding, range validation, source split consistency, lineage, and raw-only Stop.
- [x] Implement helpers and extend recording/extraction/augmentation controls; keep standalone synchronized.
- [x] Update English documentation and usage examples.
- [x] Run the full suite, notebook validation, lint/format, and independent review; report behavior and limits.

## Evidence and Decisions

- Seven new windowing tests were run before implementation and failed on missing extraction and category support. Earlier negative tests were updated to exercise the approved explicit extraction flow rather than automatic processing at Stop.
- A new hard-negative Start/Stop test exposed rounding of the default full-recording interval beyond EOF; using the exact duration and mapping EOF to the exact frame count fixed it.
- Full suite: 34 tests pass, including a sparse one-hour WAV with reads limited to selected windows near its end, 44.1 kHz stereo resampling, raw-only negative Stop, padded hard negatives, and the earlier real-kernel single-output preview test.
- Notebook validation, AST equivalence of the standalone cell, lint, formatting, and whitespace checks pass. No actual microphone recording or hour-long hardware capture was performed.
- Source splitting applies to new negative extractions. The default unassigned split is locked too; cross-file speaker/session grouping and migration of previously assigned data remain explicit dataset work.

- Independent read-only review found no actionable defects and reran all 34 tests successfully. Fractional/sub-sample range quantization has implementation validation but no dedicated boundary regression test; physical microphone and VS Code frontend checks remain manual.
