import argparse
import os
import subprocess
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.experiment_sets import EXPERIMENT_SET_DEFINITIONS


def _run(cmd, cwd):
    print("[run]", " ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def main():
    parser = argparse.ArgumentParser(description="Convenience launcher for all experiment sets.")
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--aggregate-after", action="store_true")
    args = parser.parse_args()

    project_root = PROJECT_ROOT
    py = "python"

    for set_name in sorted(EXPERIMENT_SET_DEFINITIONS.keys()):
        _run(
            [
                py,
                os.path.join("scripts", "build_experiment_sets.py"),
                "--set",
                set_name,
                "--experiments-root",
                args.experiments_root,
                "--jobs-per-gpu",
                str(args.jobs_per_gpu),
            ],
            cwd=project_root,
        )
        if args.build_only:
            continue
        launch_cmd = [
            py,
            os.path.join("scripts", "launch_experiment_set.py"),
            "--set-root",
            os.path.join(args.experiments_root, set_name),
            "--jobs-per-gpu",
            str(args.jobs_per_gpu),
        ]
        if args.skip_completed:
            launch_cmd.append("--skip-completed")
        _run(launch_cmd, cwd=project_root)

        if args.aggregate_after:
            _run(
                [
                    py,
                    os.path.join("scripts", "aggregate_experiment_set.py"),
                    "--set-root",
                    os.path.join(args.experiments_root, set_name),
                ],
                cwd=project_root,
            )


if __name__ == "__main__":
    main()
