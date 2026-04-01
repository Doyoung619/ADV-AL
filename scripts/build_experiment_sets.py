import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs.experiment_sets import EXPERIMENT_SET_DEFINITIONS
from utils.experiment_naming import canonical_method_to_cli, experiment_folder_name, method_slug
from utils.gpu_scheduler import discover_gpu_ids, write_gpu_command_scripts


def _set_extra_tag(spec: Dict) -> str:
    tags = []
    if spec.get("budget_tag"):
        tags.append(spec["budget_tag"])
    if spec.get("rounds_tag"):
        tags.append(spec["rounds_tag"])
    if spec.get("epsilon_tag"):
        tags.append(spec["epsilon_tag"])
    return "_".join(tags)


def expand_set_specs(set_name: str, set_def: Dict) -> List[Dict]:
    rows: List[Dict] = []
    for dataset in set_def["datasets"]:
        for model in set_def["models"]:
            for method in set_def["methods"]:
                base = {
                    "set_name": set_name,
                    "dataset": dataset,
                    "model": model,
                    "method": method_slug(method),
                    "seeds": list(set_def["seeds"]),
                    "initial_labeled_size": int(set_def["initial_labeled_size"]),
                    "acquisition_size": int(set_def["acquisition_size"]),
                    "rounds": int(set_def["rounds"]),
                    "epochs_per_round": int(set_def["epochs_per_round"]),
                    "eval_epsilon": float(set_def["eval_epsilon"]),
                }

                variants = [base]
                if set_name == "set_D_budget_regime":
                    variants = []
                    for b in set_def["budget_regimes"]:
                        r = dict(base)
                        r["initial_labeled_size"] = int(b["initial_labeled_size"])
                        r["acquisition_size"] = int(b["acquisition_size"])
                        r["epochs_per_round"] = int(b["epochs_per_round"])
                        r["budget_tag"] = f"i{r['initial_labeled_size']}_b{r['acquisition_size']}_e{r['epochs_per_round']}"
                        variants.append(r)
                elif set_name == "set_E_round_horizon":
                    variants = []
                    for rounds in set_def["round_options"]:
                        r = dict(base)
                        r["rounds"] = int(rounds)
                        r["rounds_tag"] = f"r{int(rounds)}"
                        variants.append(r)
                elif set_name == "set_F_eval_epsilon":
                    variants = []
                    for eps in set_def["eval_epsilon_options"]:
                        r = dict(base)
                        r["eval_epsilon"] = float(eps)
                        r["epsilon_tag"] = f"eps{int(round(eps * 255.0))}by255"
                        variants.append(r)

                for r in variants:
                    extra_tag = _set_extra_tag(r)
                    exp_name = experiment_folder_name(
                        dataset=r["dataset"],
                        model=r["model"],
                        method=r["method"],
                        extra_tag=extra_tag,
                    )
                    r["experiment_name"] = exp_name
                    rows.append(r)
    rows.sort(key=lambda x: (x["dataset"], x["model"], x["method"], x.get("budget_tag", ""), x.get("rounds_tag", ""), x.get("epsilon_tag", "")))
    return rows


def _fmt_float(v: float) -> str:
    return f"{v:.10f}".rstrip("0").rstrip(".")


def build_command(
    python_exe: str,
    project_root: str,
    exp_dir: str,
    spec: Dict,
    seed: int,
    gpu_id: int,
    global_args: Dict,
) -> str:
    cli_method, extra = canonical_method_to_cli(spec["method"])
    log_path = os.path.join(exp_dir, "logs", f"seed_{seed}.log").replace("\\", "/")
    output_dir = exp_dir.replace("\\", "/")
    cmd = [
        f"CUDA_VISIBLE_DEVICES={gpu_id}",
        python_exe,
        os.path.join(project_root, "main.py").replace("\\", "/"),
        "--dataset",
        spec["dataset"],
        "--model",
        spec["model"],
        "--acquisition_method",
        cli_method,
        "--initial_labeled_size",
        str(spec["initial_labeled_size"]),
        "--acquisition_size",
        str(spec["acquisition_size"]),
        "--num_rounds",
        str(spec["rounds"]),
        "--epochs_per_round",
        str(spec["epochs_per_round"]),
        "--epsilon",
        _fmt_float(spec["eval_epsilon"]),
        "--seed",
        str(seed),
        "--output-dir",
        output_dir,
        "--run-name",
        f"seed_{seed}",
        "--num-workers",
        str(global_args["num_workers"]),
    ]
    if global_args["pin_memory"]:
        cmd.append("--pin-memory")
    else:
        cmd.append("--no-pin-memory")
    if global_args["download_if_missing"]:
        cmd.append("--download-if-missing")
    if global_args["acquisition_pool_subset_size"] is not None:
        cmd.extend(["--acquisition-pool-subset-size", str(global_args["acquisition_pool_subset_size"])])
    for k, v in sorted(extra.items()):
        cmd.extend([f"--{k.replace('_', '-')}", str(v)])
    return " ".join(cmd) + f" > {log_path} 2>&1"


def _assignment_slots(gpu_ids: List[int], jobs_per_gpu: int) -> List[int]:
    slots: List[int] = []
    for gid in gpu_ids:
        for _ in range(max(1, int(jobs_per_gpu))):
            slots.append(gid)
    return slots if slots else [0]


def write_json(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Build deterministic experiment folders/commands for AL experiment sets.")
    parser.add_argument("--set", type=str, default="all", help="Set name (e.g., set_A_main) or 'all'.")
    parser.add_argument("--experiments-root", type=str, default="experiments")
    parser.add_argument("--python-exe", type=str, default="python")
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--acquisition-pool-subset-size", type=int, default=None)
    args = parser.parse_args()

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    experiments_root = os.path.abspath(args.experiments_root)
    os.makedirs(experiments_root, exist_ok=True)

    set_names = sorted(EXPERIMENT_SET_DEFINITIONS.keys()) if args.set == "all" else [args.set]
    for s in set_names:
        if s not in EXPERIMENT_SET_DEFINITIONS:
            raise ValueError(f"Unknown set: {s}")

    gpu_ids = discover_gpu_ids()
    if len(gpu_ids) == 0:
        gpu_ids = [0]

    global_args = {
        "num_workers": int(args.num_workers),
        "pin_memory": bool(args.pin_memory),
        "download_if_missing": bool(args.download_if_missing),
        "acquisition_pool_subset_size": args.acquisition_pool_subset_size,
    }

    for set_name in set_names:
        set_def = EXPERIMENT_SET_DEFINITIONS[set_name]
        set_root = os.path.join(experiments_root, set_name)
        os.makedirs(set_root, exist_ok=True)
        write_json(os.path.join(set_root, "set_config.json"), set_def)

        specs = expand_set_specs(set_name, set_def)
        jobs = []
        exp_commands: Dict[str, List[str]] = {}
        for spec in specs:
            exp_dir = os.path.join(set_root, spec["experiment_name"])
            os.makedirs(exp_dir, exist_ok=True)
            os.makedirs(os.path.join(exp_dir, "logs"), exist_ok=True)
            write_json(os.path.join(exp_dir, "config.json"), spec)
            exp_commands[exp_dir] = []

            seed_commands = []
            for seed in spec["seeds"]:
                jobs.append((spec, int(seed), exp_dir))
                seed_commands.append(
                    {
                        "seed": int(seed),
                        "run_dir": os.path.join(exp_dir, f"seed_{seed}").replace("\\", "/"),
                        "log": os.path.join(exp_dir, "logs", f"seed_{seed}.log").replace("\\", "/"),
                    }
                )
            write_json(os.path.join(exp_dir, "seed_plan.json"), seed_commands)

        # Deterministic GPU assignment for command files.
        slots = _assignment_slots(gpu_ids=gpu_ids, jobs_per_gpu=args.jobs_per_gpu)
        commands_plain = []
        by_gpu: Dict[int, List[str]] = {g: [] for g in gpu_ids}
        for idx, (spec, seed, exp_dir) in enumerate(jobs):
            gid = slots[idx % len(slots)]
            cmd = build_command(
                python_exe=args.python_exe,
                project_root=project_root,
                exp_dir=exp_dir,
                spec=spec,
                seed=seed,
                gpu_id=gid,
                global_args=global_args,
            )
            commands_plain.append(cmd)
            by_gpu.setdefault(gid, []).append(cmd)
            exp_commands[exp_dir].append(cmd)

        for exp_dir, cmd_list in exp_commands.items():
            with open(os.path.join(exp_dir, "commands.txt"), "w", encoding="utf-8", newline="\n") as f:
                for cmd in cmd_list:
                    f.write(cmd + "\n")

        with open(os.path.join(set_root, "commands_all.txt"), "w", encoding="utf-8", newline="\n") as f:
            for cmd in commands_plain:
                f.write(cmd + "\n")

        scripts_dir = os.path.join(set_root, "commands_by_gpu")
        write_gpu_command_scripts(by_gpu, scripts_dir)
        write_json(
            os.path.join(set_root, "command_manifest.json"),
            {
                "set_name": set_name,
                "num_jobs": len(commands_plain),
                "gpu_ids": gpu_ids,
                "jobs_per_gpu": int(args.jobs_per_gpu),
                "global_args": global_args,
            },
        )

        print(f"[build] {set_name}: experiments={len(specs)} jobs={len(commands_plain)} root={set_root}")


if __name__ == "__main__":
    main()
