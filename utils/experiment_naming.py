import re
from typing import Dict, Tuple


def normalize_dataset_name(name: str) -> str:
    return name.lower().replace("-", "")


def normalize_model_name(name: str) -> str:
    return name.lower().replace("-", "")


def method_slug(name: str) -> str:
    return name.lower().replace("-", "_")


def canonical_method_to_cli(method_name: str) -> Tuple[str, Dict[str, str]]:
    """
    Map orchestration canonical names to repository CLI method + args.
    """
    m = method_slug(method_name)
    if m in {"random", "saal", "coreset", "kcenter", "core_set", "badge", "bait", "batchbald"}:
        return m, {}

    if m in {"entropy_p10_random", "entropy_p10_entropy", "bald_p10_random", "bald_p10_bald"}:
        acq = {
            "entropy_p10_random": "entropy_pfilter_random",
            "entropy_p10_entropy": "entropy_pfilter_entropy",
            "bald_p10_random": "bald_pfilter_random",
            "bald_p10_bald": "bald_pfilter_bald",
        }[m]
        return acq, {"dual_percentile": "0.1"}

    mm = re.fullmatch(r"(ours|ours_entropy|ours_bald|ours_saal|ours_badge)_p(\d+)", m)
    if mm is not None:
        head = mm.group(1)
        pct = int(mm.group(2))
        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid percentile in method name: {method_name}")
        percentile = f"{pct / 100.0:.4f}".rstrip("0").rstrip(".")
        acq = {
            "ours": "entropy_dual_b",
            "ours_entropy": "entropy_dual_b",
            "ours_bald": "bald_dual_b",
            "ours_saal": "saal_dual_b",
            "ours_badge": "badge_dual_b",
        }[head]
        return acq, {"dual_percentile": percentile}

    raise ValueError(f"Unsupported canonical method name: {method_name}")


def experiment_folder_name(
    dataset: str,
    model: str,
    method: str,
    extra_tag: str = "",
) -> str:
    ds = normalize_dataset_name(dataset)
    mdl = normalize_model_name(model)
    mth = method_slug(method)
    if extra_tag:
        return f"{ds}_{mdl}_{mth}_{extra_tag}"
    return f"{ds}_{mdl}_{mth}"
