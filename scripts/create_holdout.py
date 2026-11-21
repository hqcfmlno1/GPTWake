"""Create the GPTWake holdout from the prepared dataset batches."""

import argparse
import csv
import filecmp
import random
import shutil
import wave
from collections import defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

SEED = 42
SPEAKERS = {
    "dat": (67, 34),
    "duc": (67, 33),
    "trieuho2312": (66, 33),
}
EXTRA_HARD_HUMAN = {"dat": 16, "duc": 17, "trieuho2312": 23}
SYNTHETIC_QUOTAS = {
    "positive": {"us": 33, "gb": 33, "vn": 34},
    "hard_negative": {"us": 15, "gb": 14, "vn": 15},
}
NEGATIVE_SOURCES = (
    ("vi_vlog17", "(VLOG_#17)_", 50),
    ("en_american_accent", "How_I_Got_My_American_Accent.wav", 25),
    (
        "en_american_conversation",
        "Real_American_English_Conversation_Advanced_Listening_Practice_Master_English_Conversation_2.wav",
        25,
    ),
)


def read_batch(folder):
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["output"]
            if Path(name).name != name or Path(name).suffix.lower() != ".wav":
                raise ValueError(f"Invalid WAV filename in {folder / 'manifest.csv'}: {name}")
            yield row, folder / name


def read_ordinary_eval_windows(data):
    for stage, batch in (("processed", "batch_v2"), ("augmented", "placement_v1")):
        folder = data / "negative" / stage
        if (folder / "manifest.csv").is_file():
            yield from (entry for entry in read_batch(folder) if entry[0]["batch"] == batch)
        else:
            yield from read_batch(folder / batch)


def sample_windows(groups, count, rng, description):
    if len(groups) < count:
        raise ValueError(f"{description} requires {count} distinct inputs, found {len(groups)}.")
    return [rng.choice(sorted(groups[key])) for key in rng.sample(sorted(groups), count)]


def remaining_groups(groups, selected):
    return {source: waves for source, waves in groups.items() if selected.isdisjoint(waves)}


def select_holdout(data, rng):
    outputs = {"positive": {}, "negative": {}}
    synthetic = {}
    for index, category in enumerate(("positive", "hard_negative")):
        destination = "positive" if index == 0 else "negative"
        by_speaker = {speaker: defaultdict(list) for speaker in SPEAKERS}
        by_locale = {locale: defaultdict(list) for locale in SYNTHETIC_QUOTAS[category]}
        synthetic[category] = by_locale
        for row, path in read_batch(data / category / "augmented/batch_v1"):
            if row["context"] or row["noise"]:
                continue
            prefix, _, suffix = Path(row["source"]).name.casefold().partition("_")
            if row["source_type"] == "human" and prefix in by_speaker:
                speaker = prefix
                by_speaker[speaker][row["source"]].append(path)
            elif row["source_type"] == "synthetic" and prefix == "tts-v1":
                locale = suffix.partition("_")[0]
                if locale in by_locale:
                    by_locale[locale][row["source"]].append(path)
        for speaker, counts in SPEAKERS.items():
            for path in sample_windows(by_speaker[speaker], counts[index], rng, speaker):
                outputs[destination][f"{category}_{speaker}_{path.name}"] = path
        if category == "hard_negative":
            hard_speakers = by_speaker

    negatives = list(read_ordinary_eval_windows(data))
    original_names = {path: row.get("original_output") or path.name for row, path in negatives}
    ordinary_groups = {}
    for prefix, source_prefix, count in NEGATIVE_SOURCES:
        sources = {
            row["source"]
            for row, _ in negatives
            if Path(row["source"]).name.startswith(source_prefix)
        }
        if len(sources) != 1:
            raise ValueError(
                f"Expected one raw source matching {source_prefix}, found {len(sources)}."
            )
        source = sources.pop()
        groups = defaultdict(list)
        for row, path in negatives:
            if row["source"] == source:
                parent = Path(row.get("source_processed", path)).name
                groups[parent].append(path)
        ordinary_groups[prefix] = groups
        for path in sample_windows(groups, count, rng, source):
            outputs["negative"][f"ordinary_negative_{prefix}_{original_names[path]}"] = path

    # Draw the expansion after the original selection so existing WAVs remain identical.
    selected = set(outputs["negative"].values())
    for prefix, _, count in NEGATIVE_SOURCES:
        groups = remaining_groups(ordinary_groups[prefix], selected)
        for path in sample_windows(groups, count, rng, prefix):
            outputs["negative"][f"ordinary_negative_{prefix}_{original_names[path]}"] = path
    for speaker, count in EXTRA_HARD_HUMAN.items():
        groups = remaining_groups(hard_speakers[speaker], selected)
        for path in sample_windows(groups, count, rng, speaker):
            outputs["negative"][f"hard_negative_{speaker}_{path.name}"] = path
    for category, quotas in SYNTHETIC_QUOTAS.items():
        destination = "positive" if category == "positive" else "negative"
        for locale, count in quotas.items():
            groups = synthetic[category][locale]
            for path in sample_windows(groups, count, rng, f"{category} {locale} TTS"):
                outputs[destination][f"{category}_synthetic_{locale}_{path.name}"] = path

    if {label: len(files) for label, files in outputs.items()} != {
        "positive": 300,
        "negative": 400,
    }:
        raise ValueError("Expected 300 positive WAVs and 400 negative WAVs.")
    return outputs


def write_holdout(data, outputs, *, split="holdout"):
    destination = data / split
    expected = {
        Path(label) / name: path for label, files in outputs.items() for name, path in files.items()
    }
    existing = set()
    extend = destination.exists()
    if extend:
        actual = {path.relative_to(destination) for path in destination.rglob("*")}
        directories = {Path(label) for label in outputs}
        existing = actual - directories
        if (
            destination.is_symlink()
            or any(
                not (destination / d).is_dir() or (destination / d).is_symlink()
                for d in directories
            )
            or any(
                path not in expected
                or not (destination / path).is_file()
                or (destination / path).is_symlink()
                or not filecmp.cmp(expected[path], destination / path, shallow=False)
                for path in existing
            )
        ):
            raise FileExistsError(f"{destination} contains different files. Move it first.")
    missing = expected.keys() - existing
    if not missing:
        return

    with TemporaryDirectory(prefix=f".{split}-", dir=data) as temp:
        staging = Path(temp) / split
        for label in outputs:
            (staging / label).mkdir(parents=True)
        for relative in sorted(missing):
            path = expected[relative]
            with wave.open(str(path), "rb") as wav:
                layout = (
                    wav.getnframes(),
                    wav.getframerate(),
                    wav.getnchannels(),
                    wav.getsampwidth(),
                )
            if layout != (40000, 16000, 1, 2):
                raise ValueError(f"Expected 2.5-second mono 16 kHz PCM16 WAV: {path}")
            shutil.copyfile(path, staging / relative)
        if extend:
            for relative in sorted(missing):
                (destination / relative).hardlink_to(staging / relative)
        else:
            if destination.exists():
                raise FileExistsError(f"{destination} was created during the run.")
            staging.rename(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    try:
        outputs = select_holdout(args.data_dir, random.Random(SEED))
        write_holdout(args.data_dir, outputs)
    except (OSError, ValueError, KeyError, wave.Error) as error:
        parser.exit(1, f"Could not create holdout: {error}\n")
    print(f"{args.data_dir / 'holdout'}: 300 positive, 200 ordinary negative, 200 hard negative.")


if __name__ == "__main__":
    main()
