# Simplify Negative Preparation Controls

The user found the negative preparation UI confusing after adding a slider. The slider and textual Ranges were competing sources of truth, irrelevant parameters stayed visible, and a short hard-negative recording could be submitted in random-window mode without actionable guidance.

## Design

- Use the slider directly for extraction by default. Source selection initially selects the whole recording; no Use range click is needed.
- Hard-negative sources default to keeping a selected clip and padding it to the chosen output window. Ordinary negatives default to random windows.
- Show sample count only for random sampling and stride only for sliding. Show an output-window selector in the negative section linked to the existing window control.
- Put multiple intervals and source split assignment in a collapsed Advanced accordion. Enable multiple intervals explicitly; otherwise the slider remains authoritative. Retain replace/add operations only inside the multiple-interval panel.
- Show a live, short explanation of the selected duration, output duration, and padding or sampling behavior. Disable extraction for invalid selections and explain the correction before submission.
- Separate Save samples from Augment saved samples, explicitly stating that augmentation applies to every saved sample in the chosen category. Move legacy import under a separate collapsed tools panel.
- Preserve recording, labeling, storage, split lock, preview publication, and standalone-cell behavior.

## Verification

- [x] Failing regressions for direct slider extraction, hard-negative default padding, mode-sensitive controls, multiple-interval opt-in, short-window guidance, and synchronized window selectors.
- [x] Implement controls and update English usage documentation.
- [x] Full tests, notebook validation and standalone equivalence, lint/format, and final review.

## Verification Evidence

- Four new behavior tests failed before implementation: direct slider extraction used stale Ranges, source changes retained an unsuitable random mode, irrelevant count/stride controls stayed visible, and short selections had no preflight guidance. All now pass.
- Full suite: 41 tests pass, including hard-negative 1.33-second selection padded to 2 seconds, direct slider range lineage, explicit multiple-interval extraction, linked output duration controls, prior streaming/long-file extraction tests, and real-kernel single preview publication.
- Notebook structure, clean outputs, standalone AST equivalence, lint, formatting, and whitespace checks pass. Live VS Code rendering and physical microphone remain untested.

- Independent review identified a still-editable source split after its first save. Added a regression, observed it fail, then locked the selector and displayed its assigned split. The complete suite now passes 42 tests; lint/format and notebook equivalence remain clean.

## Correction: Hard-Negative Random Placement

The user clarified that hard-negative random sampling means placing the complete processed event at random positions inside the output window. The earlier hard-negative random option cropped full windows from a longer source, which contradicted that expectation.

- Hard-negative UI now exposes random placement and centered placement only; ordinary negative random cropping/sliding stays separate. New hard-negative sources default to one randomly placed copy; Copies / clip controls the number for each approved clip.
- The helper reads/normalizes each selected clip once, then creates the requested windows without truncating its content. Placement offsets and the seed are stored in per-window metadata and inherited by augmentation. Positions are sampled independently, so repeats are allowed. Exact-fit clips have offset zero.
- Added regressions first for one/twenty random copies with sample-content preservation, oversized/invalid counts, exact fits, multiple clips, and a 2.5-second output in the UI. Tests initially failed on the missing placement mode; they pass after implementation.
