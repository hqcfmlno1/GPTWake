"""Create one reproducible GPTWake training split."""

import argparse
import csv
import filecmp
import json
import os
from collections import defaultdict
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory

SEED = 42
POSITIVE_COUNT = 25_000
NEGATIVE_COUNT = 21_400
PRESETS = {"1": 0, "2": 25, "3": 50, "4": 70}
EVALUATION_COUNTS = {"holdout": (300, 200, 200), "validation": (100, 50, 50)}


def read_batch(folder):
    with (folder / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            name = row["output"]
            if Path(name).name != name or Path(name).suffix.lower() != ".wav":
                raise ValueError(f"Invalid WAV filename in {folder / 'manifest.csv'}: {name}")
            yield row, folder / name


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


def evaluation_index(batches):
    index = {category: defaultdict(list) for category in batches}
    for category, rows in batches.items():
        for row, path in rows:
            if category == "ordinary_negative":
                # The published ordinary negatives came from these two original batches.
                if row.get("batch") not in (
                    "batch_v2",
                    "placement_v1",
                ) and path.parent.name not in (
                    "batch_v2",
                    "placement_v1",
                ):
                    continue
            name = row.get("original_output") or path.name
            index[category][name].append((row, path))
    return index


def evaluation_rows(data, split, index):
    selected = {category: [] for category in index}
    for category, count in zip(index, EVALUATION_COUNTS[split], strict=True):
        label = "positive" if category == "positive" else "negative"
        published = sorted((data / split / label).glob(f"{category}_*.wav"))
        if len(published) != count:
            raise ValueError(f"Expected {count} {category} WAVs in {split}.")
        for file in published:
            # Evaluation filenames prepend the label and group to the original output name.
            parts = file.name.removeprefix(f"{category}_").split("_")
            candidates = [
                entry
                for pos in range(1, len(parts))
                for entry in index[category].get("_".join(parts[pos:]), ())
            ]
            check_wav(file)
            matches = [
                (row, path) for row, path in candidates if filecmp.cmp(file, path, shallow=False)
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Could not match {file} to one source WAV. Check the LFS checkout."
                )
            selected[category].append(matches[0])
        if len({path for _, path in selected[category]}) != count:
            raise ValueError(f"Duplicate source WAVs in {split} {category}.")
    return selected


def rank_samples(data, paths, category):
    def key(path):
        relative = path.relative_to(data).as_posix()
        return sha256(f"{SEED}:{category}:{relative}".encode()).digest(), relative

    return sorted(set(paths), key=key)


def training_pools(data):
    batches = {
        "positive": list(read_batch(data / "positive/augmented/batch_v1")),
        "ordinary_negative": list(ordinary_windows(data)),
        "hard_negative": list(read_batch(data / "hard_negative/augmented/batch_v1")),
    }
    index = evaluation_index(batches)
    holdout = evaluation_rows(data, "holdout", index)
    validation = evaluation_rows(data, "validation", index)
    held_recordings = {row["source"] for row, _ in holdout["ordinary_negative"]}
    valid_recordings = {row["source"] for row, _ in validation["ordinary_negative"]}
    evaluation_recordings = held_recordings | valid_recordings
    valid_parents = {
        negative_parent(data, row, path) for row, path in validation["ordinary_negative"]
    }
    if (len(held_recordings), len(valid_parents)) != (3, 50):
        raise ValueError("Could not match the fixed evaluation negatives to their source windows.")
    pools = {
        "ordinary_negative": [
            path
            for row, path in batches["ordinary_negative"]
            if row["source"] not in held_recordings
            and negative_parent(data, row, path) not in valid_parents
        ]
    }
    held_speakers = {
        Path(row["source"]).name.partition("_")[0].casefold()
        for category in ("positive", "hard_negative")
        for row, _ in holdout[category]
        if row["source_type"] == "human"
    }
    for category in ("positive", "hard_negative"):
        held_sources = {row["source"] for row, _ in holdout[category] + validation[category]}
        eligible = []
        for row, path in batches[category]:
            source = row["source"]
            speaker = Path(source).name.partition("_")[0].casefold()
            if source in held_sources or (
                row["source_type"] == "human" and speaker in held_speakers
            ):
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
