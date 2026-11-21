"""Create the fixed GPTWake validation WAVs from the prepared batches."""

import argparse
import random
import wave
from collections import defaultdict
from pathlib import Path

from create_holdout import SEED, read_batch, sample_windows, select_holdout, write_holdout

PHRASE_QUOTAS = {
    "positive": {"duong": 25, "hanhh": 25, "tts_us": 17, "tts_gb": 16, "tts_vn": 17},
    "hard_negative": {"duong": 13, "hanhh": 12, "tts_us": 8, "tts_gb": 8, "tts_vn": 9},
}
ORDINARY_QUOTAS = {
    "vi_haveasip": ("#HaveASip_117.wav", 25),
    "en_british_practice": ("45_minutes_of_British_English_practice.wav", 13),
    "en_listening": ("English_Listening_Practice_-_REAL_Native_Conversation!_(B2_-_C1).wav", 12),
}


def phrase_group(row):
    prefix, _, suffix = Path(row["source"]).name.casefold().partition("_")
    return f"tts_{suffix.partition('_')[0]}" if row["source_type"] == "synthetic" else prefix


def select_validation(data):
    rng = random.Random(SEED)
    holdout_paths = {
        path
        for files in select_holdout(data, random.Random(SEED)).values()
        for path in files.values()
    }
    outputs = {"positive": {}, "negative": {}}
    for category, quotas in PHRASE_QUOTAS.items():
        batch = list(read_batch(data / category / "augmented/batch_v1"))
        held_sources = {row["source"] for row, path in batch if path in holdout_paths}
        groups = {group: defaultdict(list) for group in quotas}
        for row, path in batch:
            group = phrase_group(row)
            source = row["source"]
            if (
                group in groups
                and source not in held_sources
                and not (row["noise"] or row["context"])
            ):
                groups[group][source].append(path)
        target = outputs["positive" if category == "positive" else "negative"]
        for group, count in quotas.items():
            for path in sample_windows(groups[group], count, rng, f"{category} {group}"):
                target[f"{category}_{group}_{path.name}"] = path

    batches = ("negative/processed/batch_v2", "negative/augmented/placement_v1")
    ordinary = [entry for batch in batches for entry in read_batch(data / batch)]
    held_sources = {row["source"] for row, path in ordinary if path in holdout_paths}
    for group, (suffix, count) in ORDINARY_QUOTAS.items():
        matches = [
            (row, path) for row, path in ordinary if Path(row["source"]).name.endswith(suffix)
        ]
        sources = {row["source"] for row, _ in matches}
        if len(sources) != 1 or sources & held_sources:
            raise ValueError(f"Expected one source outside the holdout for {group}.")
        parents = defaultdict(list)
        for row, path in matches:
            parent = Path(row.get("source_processed") or path).name
            parents[parent].append(path)
        for path in sample_windows(parents, count, rng, group):
            outputs["negative"][f"ordinary_negative_{group}_{path.name}"] = path

    if {label: len(files) for label, files in outputs.items()} != {
        "positive": 100,
        "negative": 100,
    }:
        raise ValueError("Expected 100 positive WAVs and 100 negative WAVs.")
    return outputs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=Path(__file__).resolve().parents[1] / "data"
    )
    args = parser.parse_args()
    try:
        data = args.data_dir.resolve()
        write_holdout(data, select_validation(data), split="validation")
    except (OSError, ValueError, KeyError, wave.Error) as error:
        parser.exit(1, f"Could not create validation: {error}\n")
    print(f"{data / 'validation'}: 100 positive, 50 ordinary negative, 50 hard negative.")


if __name__ == "__main__":
    main()
