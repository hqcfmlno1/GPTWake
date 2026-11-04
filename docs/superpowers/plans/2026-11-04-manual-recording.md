# Manual Recording and Preview Implementation Plan

**Goal:** Resolve duplicate preview output and let users explicitly start and stop every recording category, without countdown or automatic wake-word boundaries.

**Spec:** The user's request in this conversation: investigate duplicated previews for all six variants, remove the three-second countdown and heuristic detection, and use the noise-style Start/Stop flow for recording.

**Architecture:** Reuse the streaming recorder for positive, negative, and noise. Use one active recording at a time. New positive selections cover the complete recording; retain optional manual trimming and reuse only explicitly marked manual selections. Keep the three original code cells and the independent final cell synchronized. Route callback logs and previews directly to Output widget state, avoiding frontend message capture, and dispose the previous panel's child widgets on rerun.

**Tech Stack:** Python, sounddevice, SoundFile, ipywidgets, audiomentations, unittest, Jupyter kernel protocol.

## Constraints and Review Focus

- Keep all raw and existing generated data, including the user's removal of obsolete placeholder files.
- All Markdown content stays in English; do not commit or push.
- No real microphone recording in automated checks.
- Start/Stop controls must remain responsive; failed/empty/active recordings cannot enter processing.
- No automatic trimming, including reuse of old unmarked heuristic intervals.
- Positive selections must still fit the chosen augmentation window; negative recordings shorter than a window stay saved as raw with an explanation.
- Each preview must be emitted once, with no second stream/display publication path; reruns must not keep active old controls.

## Steps

- [x] Diagnose callback count, generated files, and kernel output messages in an isolated kernel.
- [x] Add and run failing tests for manual full-recording selection, Start/Stop positive and negative capture, short/failed recordings, panel disposal, and one output entry per preview.
- [x] Implement the shared recorder, explicit widget outputs, manual-only selections, and updated controls.
- [x] Update notebook and README instructions; keep the standalone cell equivalent to the original three cells.
- [x] Run the complete tests, lint, formatting, notebook validation, kernel message checks, and review the changes.

## Evidence and Decisions

- Before the change, a fresh kernel emitted 7 audio displays and 7 SVG displays (the selected source plus six variants), one quieter log, and one output run. The duplicate screenshot therefore was not explained by duplicate augmentation calls. The exact VS Code renderer defect was not reproduced outside VS Code.
- After the change, button callbacks emit no IOPub `stream`, `display_data`, or `clear_output` messages. The Output widget state contains exactly 7 audio players, 7 waveforms, and one log per preview. This removes dependence on frontend message capture.
- The panel previously closed only its parent VBox. It now disables and closes all child controls and the Output widget on rerun.
- Selection policy: old intervals without an explicit manual marker default to the full recording because the removed heuristic may have generated them. Raw recordings and existing augmented runs are preserved.
- Implementation continues in the current dirty workspace to preserve the ongoing notebook edits. User data and unrelated placeholder deletions are left intact; no commit or push is part of this request.
- 27 automated tests pass, including a real isolated Jupyter kernel output check. Physical microphone and live VS Code rendering remain manual checks.

- Independent read-only review found no actionable defects and confirmed the 27 passing tests; frontend and hardware checks remain as stated above.
