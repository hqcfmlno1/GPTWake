# Local Datasets

Generated and derived dataset files are ignored by Git; only raw positive and raw hard-negative recordings are tracked. Organize data by its role first, then by processing stage:

```text
data/
  positive/
    raw/          Original recordings containing the complete Hey GPT phrase
    processed/    Selected, normalized human phrases before augmentation
    synthetic/    Prepared TTS phrases, treated as processed inputs
    augmented/    Fixed-window positive variants
  negative/
    raw/          Original speech or sounds without the complete wake phrase
    processed/    Fixed-window samples extracted from recorded or imported raw WAVs
  hard_negative/
    raw/          Separately recorded confusable phrases or known false triggers
    processed/    Unpadded manually cropped or prepared TTS phrases
    augmented/    Hard-negative variants
  noise/
    raw/          Original background recordings used for mixing
```

The positive synthetic input directory is included as an empty placeholder for copy-in. Other dataset directories are created when data is saved. Notebook startup does not create empty dataset directories or timestamped session folders. Recording names such as `positive_0001.wav`, `negative_0001.wav`, `hard_negative_0001.wav`, and `noise_0001.wav` advance based on existing files, including after a kernel restart.

## Audio Stages

- **Raw:** preserve source audio before downmixing, resampling, trimming, or augmentation. All microphone categories use Start/Stop buttons with no countdown or preset duration, streaming mono RF64 float WAV at the device's default capture sample rate. Imported WAV files preserve their existing bytes, channels, sample rate, and encoding.
- **Processed:** mono 16 kHz PCM16 WAV for notebook-generated audio. For positives, **Save positive sample** stores the reviewed Phrase without padding, after verifying it fits Window. Stop and opening a raw recording do not write processed audio. Synthetic teams may supply already prepared, nonempty mono 16 kHz WAV phrases directly in `positive/synthetic/`, with no raw source or JSON requirement. Ordinary negative WAVs, including TTS sources, belong in `negative/raw/` and are extracted into `negative/processed/`. The ordinary-negative workflow does not use synthetic/ or augmented/ directories. Ordinary negatives have fixed output-window length when saved. Hard negatives save an unpadded manual phrase after checking that it fits Window; prepared TTS WAVs may be copied directly into the same hard_negative/processed/ directory, with no raw or JSON requirement. Noise uses its raw source directly without a processed copy.
- **Augmented:** WAV variants derived from human processed or prepared synthetic audio, with JSON metadata. Positives receive random placement, gain, noise, and speed variants. Hard negatives use the same six-variant placement/gain/noise/speed recipe as positives while retaining label 0. Ordinary negatives stop at processed audio and have no augmentation UI. Noise augmentation is not required by the pilot.

All stages store audio as WAV. NumPy float32 arrays are used in memory during processing; this notebook does not export `.npy` files. Raw recordings are never overwritten. Processed files can be regenerated, and a reviewed positive selection replaces its previous processed phrase. Each augmentation run has a new numbered directory and records its known source, selected interval when available, input group, and transform parameters.

## Positive Workflow for Human and Synthetic Teams

1. **Human recordings:** use **Start positive / Stop positive** for any duration. Stop saves raw only and opens a preview. The full recording is initially selected; no wake-word detection or automatic trimming is applied. Set **Phrase (s)** and use **Preview selection** to review the wake phrase.
2. Click **Save positive sample** to save that interval in `positive/processed/`, normalized to mono 16 kHz PCM16. The phrase must be nonempty and fit the current Window before WAV or metadata is written. The phrase stays unpadded. Saving creates no augmented audio and selects the saved file in the augmentation list. A later save for the same raw filename replaces its processed phrase; existing augmented runs stay intact.
3. **Synthetic recordings:** generate one reviewed wake phrase per file, and copy ready-made WAVs directly into `data/positive/synthetic/`. Use nonempty mono 16 kHz audio, preferably PCM16, with the entire file fitting Window. No raw copy or JSON sidecar is needed. Imported files are not silently cropped, resampled, or modified during augmentation. Files must be directly inside synthetic; the positive input lists do not scan subfolders.
4. Use distinct names across teammates and TTS voices, such as `human_alex_001.wav` and `tts_voice01_001.wav`, to avoid overwriting files while combining datasets. Click **Refresh positive files** after copying files. Refresh preserves selections that still exist.
5. Under **Augment positive processed audio**, the default **Selected processed files** scope uses only highlighted files. Ctrl/Cmd-click selects individual files; Shift-click selects a range. **Augment all human** uses every WAV in `positive/processed/`; **Augment all synthetic** uses every WAV in `positive/synthetic/`. The file list prefixes labels with `human/` or `synthetic/`. All three scopes create six augmented variants per valid positive file and never combine the two groups implicitly. The button is disabled when no files are targeted, including when only raw recordings exist.
6. Augmentation reads prepared inputs only, from human processed or synthetic. Changing raw Phrase bounds without saving cannot affect its inputs. Invalid, empty, wrong-rate, stereo, or oversized processed files are reported and skipped; other valid files continue. Every successful click creates new numbered augmentation runs. Raw and processed inputs are preserved.

For notebook-generated positives, JSON records the raw source and manual bounds. Opening a raw source restores explicitly saved manual bounds; older intervals without a `selection_method: manual` marker default to the full recording. Existing processed files are preserved when a new selection is rejected. External prepared inputs without JSON use their actual file path as the known source and mark `source_stage: processed`; speaker and split default to `unassigned`. Augmented metadata records `source_type: human` or `source_type: synthetic` from the selected input folder. Synthetic augmentation run names have a `synthetic_` prefix, so same-name files from separate groups remain distinguishable. Optional sidecars can provide speaker, split, TTS model/voice, and provenance; supplied metadata is carried into augmented runs. Do not infer that metadata-free audio has a known human speaker or raw recording.

## Ordinary Negatives

Ordinary negatives cover everyday speech, silence, and environmental sounds without the complete wake phrase. Their label is **0**.

1. Use **Start negative / Stop negative** for microphone recordings. Every Stop finalizes a raw WAV immediately. Alternatively, copy external WAVs (any sample rate, mono or multichannel) into `negative/raw/`, then click **Refresh raw WAVs**. Imported raw WAVs need no JSON.
2. Select **Category: Negative** and **Process: Selected raw WAV**, choose a Source, drag **Selection (s)**, and click **Preview selection** to listen before saving. Only the chosen interval is decoded for preview.
3. Set **Output (s)**, linked to the positive Window selector. **Random crops from recording** requests **Samples** unique starting positions; windows may overlap. **Slide through selection** uses **Stride (s)** and omits incomplete tails. Each approved interval must fit a complete output window.
4. Click **Save samples** to save fixed-window mono 16 kHz PCM16 WAVs in `negative/processed/`. There is no ordinary-negative augmentation UI.
5. **All raw WAVs (full duration per file)** ignores the slider and interval list and processes every full WAV in `negative/raw/`. Random mode requests Samples per file; sliding applies Stride independently to each file. Click **Process all raw WAVs**. Invalid, active, failed, empty, or short files are reported and skipped. Existing split locks are preserved; unlocked sources use the chosen split.
6. For selected-file processing, **Advanced: dataset split and several intervals** supports a split and several non-overlapping approved intervals. Enabling **Use several time intervals** replaces the slider with the editable interval list. The first extraction locks the source split, including `unassigned`; changing it later requires coordinated migration.

Each extraction creates a new folder such as `negative/processed/negative_0001_extract_0001/window_0001.wav`, per-window JSON, and an `extraction.json` manifest. Window extraction reads one segment at a time without loading a full long recording. Source audio is preserved.

## Hard Negatives

Hard negatives are confusable partial phrases, similar-sounding words, or known false triggers. They use label **0** and follow the positive phrase workflow rather than ordinary-negative random/sliding extraction.

1. Use **Start hard negative / Stop hard negative** for any duration. Stop saves only `hard_negative/raw/`; opening a recording or previewing it does not save processed audio.
2. Select **Category: Hard negative**, choose a raw recording, crop **Phrase (s)** manually, and use **Preview selection**. The preview appears beside the phrase controls. There is no automatic wake-word detection, random cropping, or window placement at this stage.
3. Click **Save hard-negative sample**. The crop is decoded, downmixed and resampled to mono 16 kHz PCM16, then saved directly as `hard_negative/processed/<raw-stem>.wav`, without padding. Empty or oversized phrases are rejected before any existing processed audio is overwritten. Reopening the source restores saved manual bounds. Saving the same raw source again replaces that source's processed phrase; existing augmented runs remain.
4. For TTS, copy reviewed, nonempty mono 16 kHz WAV phrases directly into `hard_negative/processed/`. The whole phrase must fit Window. Raw audio and JSON sidecars are optional for these prepared inputs. Use distinct filenames across speakers, voices, and teammates. Optional JSON may provide `source_type: synthetic`, model, voice, speaker, and split; metadata-free imported audio has an unassigned source type rather than an inferred human origin.
5. Click **Refresh file list**, highlight individual files, or choose **Augment all processed**. Recorded crops and copied TTS WAVs share this list. A manual save selects its output automatically. Ctrl/Cmd-click selects several files; Shift-click selects a range.
6. Augmentation validates format, nonempty finite audio, metadata if supplied, and duration against the current Window for each file. Invalid files are reported and skipped so valid files continue. It reads only saved processed phrases; unsaved slider edits and raw recordings do not affect augmentation.
7. Each valid phrase produces **six fixed-window variants**: three random placements, lower gain at a random position, added noise at a random position, and slightly faster speech at a random position. Silence fills unused positions. Saved variants live in `hard_negative/augmented/`, with label 0 and source metadata. Exact-fit phrases cannot move; oversized phrases are never silently truncated.

Previously saved hard-negative extraction folders remain selectable for compatibility; they are not automatically trimmed or rewritten. New manual saves use unpadded phrases at the processed root. Keep all descendants of a source in its original dataset split. **Legacy data tools** contains the old-layout import button.

## Labels and Source Tracking

A positive mixed with noise remains positive; a negative mixed with noise remains negative. Noise tracks are sources for mixing and do not automatically become negative training examples. To use a noise segment as a negative example, explicitly prepare and label it with its original source information.

JSON sidecars identify `recording_id`, `source`, `category`, `label`, `speaker_id`, `split`, sample rate, and selected intervals where applicable. Extracted ordinary-negative windows also store `source_group`, exact source-frame bounds and rate, padding, windowing mode, and random seed. Manually saved hard-negative phrases store manual bounds, source-frame bounds, and provenance; augmentation records placement offsets in seconds. Augmentation metadata carries this lineage plus the processed source and actual transform parameters. Add accent, transcript, recording-session groups, and annotation review status when collecting a formal dataset.

Split speakers and recording sessions before generating a real train/validation/test dataset. Source split locking prevents new extractions of one raw file from crossing splits; it does not automatically group different recordings from the same speaker/session or repair older dataset assignments. All crops and augmented descendants of a source must remain in its split. Noise used for training augmentation must also respect the planned source separation; the pilot's shared noise pool is not a split-aware training pipeline.

## Reusing Noise and Legacy Data

Noise files under `noise/raw/` are shared across notebook restarts. Click **Start noise** to record without a preset time limit and **Stop noise** to finalize the recording. Audio blocks stream to disk as RF64 float WAV at the device sample rate, allowing files larger than 4 GiB without keeping the whole recording in memory. Keep the kernel running until Stop finishes. Device or disk errors can also end capture; failed, active, and empty recordings are excluded from mixing. Playback previews only the first five seconds.

For each noise variant, the mixer randomly chooses an eligible file, then a start offset that fits the required window. Only that segment is decoded, downmixed, and resampled as needed. Noise level is sampled at 10-25 dB SNR relative to the input window. Tracks shorter than the window repeat from their beginning. Metadata records the file, offset, requested duration, and SNR. Without eligible files, the pilot falls back to Gaussian noise. Noise does not use a processed/ directory.

The notebook's **Import old recordings** button copies sources from the previous `raw/positive_pilot/` and `negatives/room_noise_pilot/` layouts into the new categories. Original files are preserved, and recorded `legacy_source` paths prevent duplicate imports. Already-resampled legacy WAV files retain their existing format; their original device-rate signal cannot be recovered.

Do not put API keys, credentials, or tokens in metadata. Listen to positive selections and verify negative labels before training.
