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
    if m in {
        "random",
        "saal",
        "coreset",
        "kcenter",
        "core_set",
        "badge",
        "bait",
        "batchbald",
        "adv_grad_displacement_logdet",
        "adv_q_topk",
        "adv_q_filter_logdet",
        "ours_secant_badge",
        "ours_secant_logdet_refine",
        "logdet_adv_disp",
        "logdet_adv_disp_swap",
        "semantic_logdet",
        "semantic_logdet_swap",
        "adv_displacement_logdet",
        "adv_displacement_logdet_swap",
        "logdet_adv_feat_swap",
        "logdet_adv_logit_swap",
    }:
        return m, {}
    mm_adv_grad = re.fullmatch(r"adv_grad_displacement_logdet_p(\d+)", m)
    if mm_adv_grad is not None:
        pct = int(mm_adv_grad.group(1))
        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid percentile in method name: {method_name}")
        percentile = f"{pct / 100.0:.4f}".rstrip("0").rstrip(".")
        return "adv_grad_displacement_logdet", {"adv_grad_displacement_percentile": percentile}
    mm_secant_prefilter = re.fullmatch(r"ours_secant_logdet_refine_p(\d+)", m)
    if mm_secant_prefilter is not None:
        pct = int(mm_secant_prefilter.group(1))
        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid percentile in method name: {method_name}")
        return "ours_secant_logdet_refine", {"prefilter_metric": "D", "prefilter_drop_percent": str(pct)}
    mm_logdet = re.fullmatch(
        r"((?:logdet_adv_disp|semantic_logdet|adv_displacement_logdet)(?:_swap)?|logdet_adv_feat_swap|logdet_adv_logit_swap)_p(\d+)",
        m,
    )
    if mm_logdet is not None:
        head = mm_logdet.group(1)
        pct = int(mm_logdet.group(2))
        if pct < 0 or pct > 100:
            raise ValueError(f"Invalid percentile in method name: {method_name}")
        percentile = f"{pct / 100.0:.4f}".rstrip("0").rstrip(".")
        return head, {"logdet_adv_disp_percentile": percentile}

    mm_adv_q = re.fullmatch(r"adv_q_filter_logdet_([0-9]+)", m)
    if mm_adv_q is not None:
        pct = int(mm_adv_q.group(1))
        if pct <= 0 or pct > 100:
            raise ValueError(f"Invalid retain fraction in method name: {method_name}")
        retain_fraction = f"{pct / 100.0:.4f}".rstrip("0").rstrip(".")
        return "adv_q_filter_logdet", {"retain_fraction": retain_fraction}
    if m == "ours_l2":
        return "ours", {}

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
