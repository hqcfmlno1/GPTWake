# GPTWake

Local wake-word detection for **Hey GPT**, built with PyTorch. The first model
will be a reproducible BC-ResNet-8 baseline. DMA-KWS and XLS-R 300M are separate
experimental alternatives.

## Environment

This project uses **uv** and Python **3.12**. Install uv using the
[official instructions](https://docs.astral.sh/uv/getting-started/installation/),
then run:

```bash
uv sync
uv sync --all-extras
```

The first command installs the baseline dependencies and development tools.
The second also installs the optional `xlsr` and `voice` dependencies. To select
just one optional group, use `uv sync --extra voice` or `uv sync --extra xlsr`.
Commit `uv.lock`; use `uv sync --locked --all-extras` to reproduce the complete
environment. PyTorch and torchaudio use the official CUDA 12.8 wheel index for
GPU training on NVIDIA hardware. This setup targets the current Linux/Windows
NVIDIA development environment; a CPU-only or macOS installation needs a
different PyTorch source configuration. The NVIDIA driver must support CUDA
12.8 or newer; a separate CUDA toolkit is not required for normal wheel usage.
The current development machine has an RTX 3060 with 6 GB VRAM. Runtime
benchmarks will still evaluate local CPU inference.

If uv was bootstrapped locally during setup, use `.tools/bin/uv` in place of
`uv`; this local binary is ignored by Git.

```bash
uv run ruff check .
uv run ruff format --check .
```

Experiments and manual checks live in `notebooks/`. Launch JupyterLab with the
project interpreter:

```bash
uv run --all-extras jupyter lab notebooks
```

For local dataset recording and augmentation, open
`dataset_recording_augmentation.ipynb`. Its buttons capture positive, negative,
hard-negative, and background recordings into
`data/<positive|negative|hard_negative|noise>/raw/`, preserving the capture sample rate in
float WAV files. Processed audio is mono 16 kHz PCM16 WAV; augmented examples
are stored separately. Positive variants use random placement, gain, noise,
and speed changes. Each category has **Start** and **Stop** buttons;
audio streams to disk without a countdown or preset duration. Stop saves raw only.
For human positives, preview the raw recording, set **Phrase (s)**, then click
**Save positive sample**. This saves only the reviewed processed phrase after
checking it fits Window. For synthetic positives, copy prepared WAV phrases directly
into `data/positive/synthetic/`: mono 16 kHz, nonempty, and no longer than Window.
Use distinct filenames across teammates and voices; JSON sidecars are optional.
Click **Refresh positive files**, then choose individual prepared files,
**Augment all human** (from `positive/processed/`), or **Augment all synthetic**
(from `positive/synthetic/`). Each valid positive creates six variants. These scopes
never read raw or use unsaved Phrase edits; augmentation is disabled when its
selected group is empty. Invalid processed files are reported and
skipped so other valid files can continue. Positive processed phrases are not padded
until augmentation; there is no automatic wake-word detection.
Mixing randomly selects a raw
noise file and a segment within it, without creating processed noise copies.
For ordinary and hard negatives, Stop saves raw only. Ordinary negatives can also
be imported by copying WAVs with any sample rate and channel count into
`data/negative/raw/`, then clicking **Refresh raw WAVs**. Choose **Selected raw WAV**
to preview and crop approved ranges, or **All raw WAVs (full duration per file)**
to process every full WAV. Random sampling creates the requested number per file;
sliding uses the chosen stride and drops incomplete tails. Outputs are fixed-window,
mono 16 kHz PCM16 WAVs in `negative/processed/`, ready for training. Ordinary
negatives have no augmentation UI. Batch processing reports invalid, incomplete,
and short files and continues with valid files; existing source split locks remain.
Hard negatives follow a phrase workflow like positives: **Start hard negative / Stop
hard negative** saves raw only. Select **Hard negative**, preview and crop **Phrase
(s)**, then click **Save hard-negative sample**. This saves an unpadded mono 16 kHz
phrase in `hard_negative/processed/`, after checking that it fits Window. TTS teams
can copy prepared mono 16 kHz WAV phrases directly into the same processed folder;
no raw recording or JSON is required. Use distinct names to avoid overwriting
another teammate's work. **Refresh file list**, select files or **Augment all
processed**, then augment. Each valid phrase creates six fixed-window variants
with random placement, gain, noise, and speed changes, retaining label 0. Invalid
inputs are reported and skipped. Augmentation uses saved phrases and the current
Window, never unsaved crop edits. See `data/README.md` for details.
Noise is shared across notebook restarts, and folders are created only when data is saved.
The notebook uses `audiomentations` and `ipywidgets` from the development group.
The notebook contains a single standalone code cell with all setup, helpers, and
controls. Run it in the project kernel to initialize the panel without starting
microphone capture. See `data/README.md` for the recording and augmentation workflow.

Start with `00_environment.ipynb` for local GPU/audio checks, then
`01_realtime_api.ipynb` for text-to-voice and microphone-to-voice API experiments.
The Realtime notebook asks for an API key using a hidden prompt. ChatGPT Plus
alone does not provide Realtime API access; configure API billing/credits and
create a project key first. Each demo call opens a new API conversation; this
does not yet implement boot-scoped conversation persistence or full-duplex calls.
API cells are opt-in and incur usage charges. Clear notebook outputs before
committing recordings or transcripts.

## Layout

```text
configs/                 Experiment configurations (not finalized yet)
data/
  positive/              Positive raw/processed/synthetic/augmented audio
  negative/              Ordinary negative raw/processed audio
  hard_negative/         Hard-negative raw/processed/augmented audio
  noise/raw/             Background sources for mixing
src/gptwake/
  data/                  Decoding, preprocessing, windowing, augmentation
  models/                BC-ResNet, DMA-KWS adapters, XLS-R adapters
  training/              Training and fine-tuning
  evaluation/            Clip and streaming evaluation
  runtime/               Capture, detector policy, boot-scoped conversations
  integrations/          Voice API clients and authentication
scripts/                 Dataset preparation and experiment entry points
notebooks/               Environment checks and interactive experiments
docs/                    Dataset and experiment documentation
checkpoints/             Local model weights (ignored)
outputs/                 Local runs, logs, and benchmark artifacts (ignored)
```

## Dependency groups

- Baseline: PyTorch, torchaudio, NumPy, SciPy, SoundFile, sounddevice, PyYAML,
  scikit-learn, and tqdm.
- `xlsr`: Transformers, Accelerate, and safetensors. Model weights are not
  downloaded by environment setup.
- `voice`: OpenAI SDK with Realtime support and keyring. API credentials and
  API usage are separate from installing these packages.
- Development: JupyterLab, ipykernel, nbformat, nbclient, and Ruff.

DMA-KWS-specific dependencies will be added after its upstream implementation
and checkpoint requirements have been verified.

## Scope and data rules

- Binary detection of the complete English phrase **Hey GPT**.
- Keep audio window, inference hop, frontend, and decision-policy parameters
  configurable. No example duration or threshold is a fixed requirement.
- Split by speaker and source recording before augmentation/window generation.
- Local wake detection is separate from the optional cloud voice conversation.
  Conversation history is owned by GPTWake; it does not automatically sync to
  ChatGPT web or desktop.
- The Realtime notebook is a standalone API experiment. The application does
  not yet implement login, startup registration, continuous microphone
  monitoring, conversation persistence, or any model.

## System audio

`sounddevice` uses PortAudio; `soundfile` uses libsndfile. Some Linux systems
need system packages such as `libportaudio2` and `libsndfile1`, which uv cannot
install. Microphone access also depends on the user's audio session and device
permissions. Do not commit credentials, private recordings, or model weights.
