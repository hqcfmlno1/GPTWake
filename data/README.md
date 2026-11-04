# Local Datasets

Audio and generated dataset files are ignored by Git. Organize data by its role first, then by processing stage:

```text
data/
  positive/
    raw/          Original recordings containing the complete Hey GPT phrase
    processed/    Selected, normalized phrases before augmentation
    augmented/    Fixed-window positive variants
  negative/
    raw/          Original speech or sounds without the complete wake phrase
    processed/    Numbered extraction folders containing selected windows
    augmented/    Negative variants
  hard_negative/
    raw/          Separately recorded confusable phrases or known false triggers
    processed/    Numbered extraction folders containing padded clips or windows
    augmented/    Hard-negative variants
  noise/
    raw/          Original background recordings used for mixing
  manifests/      Future split assignments and dataset indexes
```

Directories are created only when data is saved. Notebook startup does not create empty dataset directories or timestamped session folders. Recording names such as `positive_0001.wav`, `negative_0001.wav`, `hard_negative_0001.wav`, and `noise_0001.wav` advance based on existing files, including after a kernel restart.

## Audio Stages

- **Raw:** preserve source audio before downmixing, resampling, trimming, or augmentation. All microphone categories use Start/Stop buttons with no countdown or preset duration, streaming mono RF64 float WAV at the device's default capture sample rate. Imported WAV files preserve their existing bytes, channels, sample rate, and encoding.
- **Processed:** mono 16 kHz PCM16 WAV. Positives initially contain the entire recording; optional manual Phrase bounds can select a shorter interval. No speech detection or automatic boundary margin is applied. Ordinary and hard-negative windows are extracted only from explicitly selected time ranges; see the workflow below. Noise uses its raw source directly without a processed copy.
- **Augmented:** WAV variants derived from processed audio, with JSON metadata. Positives receive random placement, gain, noise, and speed variants. Both negative categories receive gain, noise, and speed variants while retaining label 0. Noise augmentation is not required by the pilot.

All stages store audio as WAV. NumPy float32 arrays are used in memory during processing; this notebook does not export `.npy` files. Raw recordings are never overwritten. Processed files can be regenerated, and a reviewed positive selection replaces its previous processed phrase. Each augmentation run has a new numbered directory and snapshots its raw source, selected interval, and transform parameters.

Positive metadata marks selections as `full_recording` or `manual`. Only explicitly marked manual bounds are reused. Older bounds without this marker might have come from the removed heuristic, so reopening or reprocessing those recordings defaults to their full duration. Adjust Phrase and use **Generate 6 variants** to save a manual selection. The selected positive interval must fit the augmentation window; the window does not control recording duration.

## Ordinary and Hard Negatives

Both categories use label **0**. Ordinary negatives cover everyday speech, silence, and environmental sounds without the complete wake phrase. Hard negatives are confusable partial phrases, similar-sounding words, or events that caused a false trigger. Category identifies collection intent; difficulty still depends on the model. Noise is a separate source for mixing, not an automatic negative label.

1. Use **Start negative / Stop negative** for long recordings or **Start hard negative / Stop hard negative** for separate examples. Stop saves raw and opens the selection controls; preview is explicit.
2. In **Prepare negative samples**, choose **Category** and **Source**. The entire recording is selected initially. Drag both handles of **Selection (s)** and click **Preview selection** to hear that exact interval. The time readout can also be edited. Only the selected interval is decoded; preview memory grows with its length.
3. Set **Output (s)** for the duration of saved samples. This is linked to the Window selector used for positives. The live summary explains the selected duration, output duration, sample count, and any silence padding. Invalid selections disable Save and explain what to change.
4. Hard-negative sources default to **Random position in output window**: normalize the entire selected event, preserve every sample, and place it at a random offset inside the output window. **Copies / clip = 1** creates one random placement; **20** creates twenty copies of the same event at independently sampled positions. Each position is sampled at the 16 kHz sample resolution, including both endpoints. Fill the remaining samples with silence. Multiple approved intervals create this number of copies per interval. Oversized clips are rejected; exact-fit clips cannot move. Positions may repeat. **Centered in output window** remains available for a single centered copy per interval. Stride is hidden for both hard-negative methods.
5. Ordinary negatives default to **Random crops from recording**. **Samples** sets the count; starting positions are unique but windows may overlap. When only a few positions fit, the count is capped accordingly. **Slide through selection** instead shows **Stride (s)** and omits partial tails. Each selected interval must fit a complete output window. Crop/sliding methods are shown only for ordinary negatives; the hard-negative UI preserves the full selected event.
6. Click **Save samples**. The slider selection is used directly: no copy-to-Ranges action is required. Every saved sample stays within its approved interval. Review selections to exclude the actual wake phrase.
7. **Advanced: dataset split and several intervals** is collapsed by default. Choose Split before the first save if needed for a formal dataset. The pilot default is `unassigned`; the first extraction locks that source to its split, including unassigned. A later change requires coordinated dataset migration.
8. For multiple intervals, enable **Use several time intervals** inside Advanced. The interval list then replaces the slider as the extraction source. **Replace interval list** sets it to the current slider interval; **Add selection to list** appends another interval. You can also edit one `start,end` pair per line. The intervals must be non-overlapping and contained in the recording. Turn the checkbox off to use the slider again.
9. Under **Augment processed audio**, the default **Selected processed files** mode augments only highlighted files. **Save samples** replaces the selection with the files just saved. Use Ctrl/Cmd-click to select individual files or Shift-click for a range, then click **Augment N selected**. File labels include source/extraction folders. **Refresh file list** reloads files and preserves valid selections. Changing Category clears the selection and returns to selected-file mode.
10. To augment every processed sample in Category across all raw sources, explicitly choose **All processed files in Category** and click **Augment all N files**. Both modes create gain/noise/speed variants, preserve each sample's saved length independently of the current Output value, and create new augmentation runs on every click. They operate on existing processed audio, including legacy negative WAVs with matching metadata. **Legacy data tools** contains the old-layout import button.

Each extraction gets a new folder, such as `negative/processed/negative_0001_extract_0001/window_0001.wav`, with per-window JSON and an `extraction.json` manifest. Another click creates a new extraction folder; files are never overwritten. Extraction reads and resamples one window at a time, including for hour-long stereo/native-rate sources. Augmentation folders include the extraction name so windows with the same local filename retain distinct lineage.

## Labels and Source Tracking

A positive mixed with noise remains positive; a negative mixed with noise remains negative. Noise tracks are sources for mixing and do not automatically become negative training examples. To use a noise segment as a negative example, explicitly prepare and label it with its original source information.

JSON sidecars identify `recording_id`, `source`, `category`, `label`, `speaker_id`, `split`, sample rate, and selected intervals where applicable. Extracted negative windows also store `source_group`, exact source-frame bounds and rate, padding, windowing mode, random seed, and hard-negative placement offset in samples and seconds. Augmentation metadata carries this lineage plus the processed source and actual transform parameters. Add accent, transcript, recording-session groups, and annotation review status when collecting a formal dataset.

Split speakers and recording sessions before generating a real train/validation/test dataset. Source split locking prevents new extractions of one raw file from crossing splits; it does not automatically group different recordings from the same speaker/session or repair older dataset assignments. All crops and augmented descendants of a source must remain in its split. Noise used for training augmentation must also respect the planned source separation; the pilot's shared noise pool is not a split-aware training pipeline.

## Reusing Noise and Legacy Data

Noise files under `noise/raw/` are shared across notebook restarts. Click **Start noise** to record without a preset time limit and **Stop noise** to finalize the recording. Audio blocks stream to disk as RF64 float WAV at the device sample rate, allowing files larger than 4 GiB without keeping the whole recording in memory. Keep the kernel running until Stop finishes. Device or disk errors can also end capture; failed, active, and empty recordings are excluded from mixing. Playback previews only the first five seconds.

For each noise variant, the mixer randomly chooses an eligible file, then a start offset that fits the required window. Only that segment is decoded, downmixed, and resampled as needed. Noise level is sampled at 10-25 dB SNR relative to the input window. Tracks shorter than the window repeat from their beginning. Metadata records the file, offset, requested duration, and SNR. Without eligible files, the pilot falls back to Gaussian noise. Earlier `noise/processed/` files are preserved but no longer used.

The notebook's **Import old recordings** button copies sources from the previous `raw/positive_pilot/` and `negatives/room_noise_pilot/` layouts into the new categories. Original files are preserved, and recorded `legacy_source` paths prevent duplicate imports. Already-resampled legacy WAV files retain their existing format; their original device-rate signal cannot be recovered.

Do not put API keys, credentials, or tokens in metadata. Listen to positive selections and verify negative labels before training.
