import argparse
import os
from typing import List


HEAVY_FILENAMES = {
    "best_model.pt",
}

OPTIONAL_REMOVE_FILES = {
    "train_history.csv",
    "train_history.json",
    "timing_log.csv",
    "timing_log.json",
}


def _iter_files(root: str):
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            yield dirpath, fn, os.path.join(dirpath, fn)


def compact_experiments(root: str, remove_training_history: bool, keep_logs: bool) -> List[str]:
    removed = []
    for dirpath, fn, path in _iter_files(root):
        if fn in HEAVY_FILENAMES:
            os.remove(path)
            removed.append(path)
            continue

        if remove_training_history and fn in OPTIONAL_REMOVE_FILES:
            os.remove(path)
            removed.append(path)
            continue

        if not keep_logs and fn.endswith(".log") and os.path.basename(dirpath) == "logs":
            os.remove(path)
            removed.append(path)
            continue
    return removed


def main():
    parser = argparse.ArgumentParser(description="Remove heavy artifacts from experiments for easier download.")
    parser.add_argument("--experiments-root", type=str, required=True)
    parser.add_argument("--remove-training-history", action="store_true")
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args()

    root = os.path.abspath(args.experiments_root)
    removed = compact_experiments(
        root=root,
        remove_training_history=bool(args.remove_training_history),
        keep_logs=bool(args.keep_logs),
    )
    print(f"[compact] root={root}")
    print(f"[compact] removed_files={len(removed)}")


if __name__ == "__main__":
    main()
