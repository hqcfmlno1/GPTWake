"""Create one reproducible GPTWake training split."""

import argparse
import filecmp
import json
import os
import random
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

from create_holdout import SEED, SPEAKERS, read_batch, select_holdout
from create_validation import select_validation

POSITIVE_COUNT = 25_000
NEGATIVE_COUNT = 21_400
PRESETS = {"1": 0, "2": 25, "3": 50, "4": 70}


def ordinary_windows(data):
    batches = {
        "processed": ("batch_v1", "batch_v2", "batch_v3"),
        "augmented": ("placement_v1", "placement_v2"),
    }
    for stage, names in batches.items():
        folder = data / "negative" / stage
        if (folder / "manifest.csv").is_file():
            yield from read_batch(folder)
        else:
            for name in names:
                yield from read_batch(folder / name)


def negative_parent(data, row, path):
    parent = row.get("source_processed")
    return parent.removeprefix("data/") if parent else path.relative_to(data).as_posix()


def evaluation_paths(data, split, outputs):
    destination = data / split
    expected = {
        destination / label / name: source
        for label, files in outputs.items()
        for name, source in files.items()
    }
    if destination.exists() and (
        {path for path in destination.glob("*/*.wav")} != expected.keys()
        or any(not filecmp.cmp(path, source, shallow=False) for path, source in expected.items())
    ):
        raise ValueError(f"{destination} differs from the fixed evaluation selection.")
    return set(expected.values())


def rank_samples(data, paths, category):
    def key(path):
        relative = path.relative_to(data).as_posix()
        return sha256(f"{SEED}:{category}:{relative}".encode()).digest(), relative

    return sorted(set(paths), key=key)


def training_pools(data):
    holdout = evaluation_paths(data, "holdout", select_holdout(data, random.Random(SEED)))
    validation = evaluation_paths(data, "validation", select_validation(data))
    ordinary = list(ordinary_windows(data))
    held_recordings = {row["source"] for row, path in ordinary if path in holdout}
    valid_recordings = {row["source"] for row, path in ordinary if path in validation}
    evaluation_recordings = held_recordings | valid_recordings
    valid_parents = {
        negative_parent(data, row, path) for row, path in ordinary if path in validation
    }
    if (len(held_recordings), len(valid_parents)) != (3, 50):
        raise ValueError("Could not match the fixed evaluation negatives to their source windows.")
    pools = {
        "ordinary_negative": [
            path
            for row, path in ordinary
            if row["source"] not in held_recordings
            and negative_parent(data, row, path) not in valid_parents
        ]
    }
    for category in ("positive", "hard_negative"):
        rows = list(read_batch(data / category / "augmented/batch_v1"))
        held_sources = {row["source"] for row, path in rows if path in holdout | validation}
        eligible = []
        for row, path in rows:
            source = row["source"]
            speaker = Path(source).name.partition("_")[0].casefold()
            if source in held_sources or (row["source_type"] == "human" and speaker in SPEAKERS):
                continue
            if any(
                clip["source"] in evaluation_recordings
                for clip in json.loads(row.get("context") or "[]")
            ):
                continue
            eligible.append((row, path))
        if category == "positive":
            human = (path for row, path in eligible if row["source_type"] == "human")
            synthetic = (path for row, path in eligible if row["source_type"] == "synthetic")
            pools[category] = rank_samples(data, human, "positive_human") + rank_samples(
                data, synthetic, "positive_synthetic"
            )
        else:
            pools[category] = rank_samples(data, (path for _, path in eligible), category)
    pools["ordinary_negative"] = rank_samples(data, pools["ordinary_negative"], "ordinary_negative")
    return pools


def select_training(data, percent):
    pools = training_pools(data)
    hard = NEGATIVE_COUNT * percent // 100
    quotas = {
        "positive": POSITIVE_COUNT,
        "ordinary_negative": NEGATIVE_COUNT - hard,
        "hard_negative": hard,
    }
    selected = {}
    for category, count in quotas.items():
        if len(pools[category]) < count:
            raise ValueError(
                f"{category} requires {count:,} trainable WAVs, found {len(pools[category]):,}."
            )
        label = "positive" if category == "positive" else "negative"
        for source in pools[category][:count]:
            suffix = "_".join(source.relative_to(data).parts[1:])
            selected[Path(label) / f"{category}_{suffix}"] = source
    if len(selected) != POSITIVE_COUNT + NEGATIVE_COUNT:
        raise ValueError("Training filenames collide.")
    return quotas, selected


def write_training(destination, selected):
    folders = {Path("positive"), Path("negative")}
    links = {
        relative: Path(os.path.relpath(source, (destination / relative).parent))
        for relative, source in selected.items()
    }
    if destination.exists() or destination.is_symlink():
        if (
            destination.is_symlink()
            or any((destination / folder).is_symlink() for folder in folders)
            or {p.relative_to(destination) for p in destination.rglob("*")}
            != links.keys() | folders
            or any(
                not (destination / name).is_symlink() or (destination / name).readlink() != target
                for name, target in links.items()
            )
        ):
            raise FileExistsError(f"{destination} differs from this selection. Move it first.")
        for source in selected.values():
            check_wav(source)
        return

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".train-", dir=destination.parent) as temp:
        staging = Path(temp) / destination.name
        for folder in folders:
            (staging / folder).mkdir(parents=True)
        for relative, source in selected.items():
            check_wav(source)
            (staging / relative).symlink_to(links[relative])
        if destination.exists() or destination.is_symlink():
            raise FileExistsError(f"{destination} was created during the run.")
        staging.rename(destination)


def check_wav(path):
    with path.open("rb") as stream:
        header = stream.read(12)
    if header[:4] not in (b"RIFF", b"RF64") or header[8:12] != b"WAVE":
        raise ValueError(
            f"Expected WAV content at {path}. Check that Git LFS files are downloaded."
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("choice", nargs="?", choices=PRESETS, help="1=H00, 2=H25, 3=H50, 4=H70")
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    choice = args.choice
    if choice is None:
        for number, percent in PRESETS.items():
            print(f"{number}. H{percent:02d} ({percent}% hard negative)")
        try:
            choice = input("Select a training split (1-4): ").strip()
        except EOFError:
            parser.error("Pass a choice from 1 to 4.")
        if choice not in PRESETS:
            parser.error("Choose a number from 1 to 4.")

    data = args.data_dir.resolve()
    percent = PRESETS[choice]
    destination = data / "train" / f"H{percent:02d}"
    try:
        quotas, selected = select_training(data, percent)
        write_training(destination, selected)
    except (OSError, ValueError, KeyError) as error:
        parser.exit(1, f"Could not create training split: {error}\n")
    counts = ", ".join(f"{count:,} {kind.replace('_', ' ')}" for kind, count in quotas.items())
    print(f"{destination} (seed {SEED}): {counts}.")


if __name__ == "__main__":
    main()
