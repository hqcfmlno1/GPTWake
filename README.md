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

For a local positive-recording pilot, open
`02_positive_recording_augmentation.ipynb`. Its buttons capture four-second
recordings, automatically save mono PCM16 WAV files, and generate randomly
positioned two-second windows with gain, noise, and speed variations.
The notebook uses `audiomentations` and `ipywidgets` from the development group.
Running all cells initializes the controls without starting microphone capture.

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
  raw/                   Original recordings
  synthetic/             Generated speech
  negatives/             Negative speech, hard negatives, and noise
  manifests/             Metadata, annotations, and split assignments
  processed/             Regenerable processed audio/features
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
