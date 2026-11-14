# Dataset Layout Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans for inline execution of this approved notebook change.

**Goal:** Store pilot data by category and processing stage without creating empty session folders.

**Architecture:** Keep the notebook's three code cells and append a standalone cell containing the same setup, helpers, and controls. The setup cell defines paths without disk writes, helper functions own storage and processing, and the control panel calls those helpers. Preserve original recordings and trace all derived files to their sources.

**Tech Stack:** Python, NumPy, SoundFile, SciPy, audiomentations, ipywidgets, unittest.

**Spec:** The approved data layout in this conversation and `data/README.md`.

## Global Constraints

- Store positive, negative, and noise sources under their own `raw` directories.
- Store normalized positive phrases and negative windows in `processed`. Use raw noise tracks directly, decoding and converting only selected segments for mixing.
- Store generated positive and negative variants in `augmented`; no noise augmentation is required.
- Raw microphone recordings preserve the device sample rate and captured float samples in WAV files. Processed and augmented audio uses mono 16 kHz PCM16 WAV.
- Use readable sequential recording names, lazy directory creation, and a shared noise library across notebook restarts.
- Keep all notebook prose and documentation in English. Do not capture real microphone audio during automated checks.

## Review Focus

- Setup reruns must not create folders or overwrite recordings.
- Native-rate and multichannel sources must survive processing unchanged.
- Cropping must preserve raw lineage and never truncate oversized positive phrases.
- Batch processing must resample native-rate sources and reuse existing noise files.
- Legacy source import must preserve old files and be idempotent.

## Task 1: Change the notebook storage contract

**Files:** Modify `notebooks/02_positive_recording_augmentation.ipynb`; create `tests/test_notebook_data_layout.py`.

- [x] Add failing behavior tests for lazy setup, native-rate preservation, category-specific processing, unique names, augmentation lineage, and legacy imports.
- [x] Run unittest discovery and confirm the old notebook fails the new storage contract.
- [x] Implement storage and processing helpers; retain random positive placement and add negative recording controls.
- [x] Update the control panel to use shared category directories and readable names.
- [x] Run behavior tests, notebook validation, lint, and formatting checks.

## Task 2: Document and preserve local data

**Files:** Modify `data/README.md`, notebook Markdown, and the relevant README paragraph.

- [x] Describe raw versus processed versus augmented audio and safe startup behavior.
- [x] Import any legacy recordings without deleting their originals, using the tested import helper.
- [x] Verify real migrated files and leave the edited notebook with cleared outputs.

## Task 3: Continuous Noise Capture and Standalone Controls

- [x] Stream noise blocks to RF64 float WAV until the Stop button is clicked, with no preset duration.
- [x] Randomly select an eligible raw noise file and a segment per noise variant; retain selection parameters in metadata.
- [x] Exclude active, failed, and empty captures and guard against replacing controls during recording.
- [x] Append an independent cell while retaining the original three code cells.
- [x] Verify 19 automated tests using simulated microphone input, including short RF64 mixing, device failure, Stop controls, and fresh-namespace initialization.
- [x] Validate the notebook, compare standalone code with the original cells, and pass repository lint and formatting checks. Physical microphone capture remains a manual check.
