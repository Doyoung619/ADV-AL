import re

from acquisition.bait import BAITStrategy
from acquisition.badge import BADGEStrategy
from acquisition.badge_dual_a import BADGEDualAStrategy
from acquisition.badge_dual_b import BADGEDualBStrategy
from acquisition.badge_adv_lagrangian import BADGEAdvLagrangianStrategy
from acquisition.badge_adv_mult import BADGEAdvMultStrategy
from acquisition.bald_adv_lagrangian import BALDAdvLagrangianStrategy
from acquisition.bald import BALDStrategy
from acquisition.batchbald import BatchBALDStrategy
from acquisition.coreset import CoreSetStrategy
from acquisition.bald_dual_a import BALDDualAStrategy
from acquisition.bald_dual_b import BALDDualBStrategy
from acquisition.entropy import EntropyStrategy
from acquisition.entropy_dual_a import EntropyDualAStrategy
from acquisition.entropy_dual_b import EntropyDualBStrategy
from acquisition.saal import SAALStrategy
from acquisition.saal_dual_b import SAALDualBStrategy
from acquisition.margin import MarginStrategy
from acquisition.margin_dual_b import MarginDualBStrategy
from acquisition.logdet_adv_disp import (
    LogDetAdvDispStrategy,
    LogDetAdvDispSwapStrategy,
    LogDetAdvFeatSwapStrategy,
    LogDetAdvLogitSwapStrategy,
)
from acquisition.ours import OursGapStrategy, OursGradDispStrategy, OursHessianStrategy, OursStrategy
from acquisition.pfilter import (
    BALDPercentileBALDStrategy,
    BALDPercentileRandomStrategy,
    EntropyPercentileEntropyStrategy,
    EntropyPercentileRandomStrategy,
)
from acquisition.random_sampling import RandomStrategy

METHOD_REGISTRY = {
    "random": RandomStrategy,
    "entropy": EntropyStrategy,
    "entropy_dual_a": EntropyDualAStrategy,
    "entropy_dual_b": EntropyDualBStrategy,
    "saal": SAALStrategy,
    "saal_dual_b": SAALDualBStrategy,
    "margin": MarginStrategy,
    "margin_dual_b": MarginDualBStrategy,
    "logdet_adv_disp": LogDetAdvDispStrategy,
    "semantic_logdet": LogDetAdvDispStrategy,
    "adv_displacement_logdet": LogDetAdvDispStrategy,
    "logdet_adv_disp_swap": LogDetAdvDispSwapStrategy,
    "logit_adv_disp_swap": LogDetAdvDispSwapStrategy,
    "semantic_logdet_swap": LogDetAdvDispSwapStrategy,
    "adv_displacement_logdet_swap": LogDetAdvDispSwapStrategy,
    "logdet_adv_feat_swap": LogDetAdvFeatSwapStrategy,
    "logit_adv_feat_swap": LogDetAdvFeatSwapStrategy,
    "logdet_adv_logit_swap": LogDetAdvLogitSwapStrategy,
    "logit_adv_logit_swap": LogDetAdvLogitSwapStrategy,
    "badge": BADGEStrategy,
    "badge_dual_a": BADGEDualAStrategy,
    "badge_dual_b": BADGEDualBStrategy,
    "badge_adv_mult": BADGEAdvMultStrategy,
    "badge_adv_lagrangian": BADGEAdvLagrangianStrategy,
    "bait": BAITStrategy,
    "bald": BALDStrategy,
    "batchbald": BatchBALDStrategy,
    "entropy_pfilter_random": EntropyPercentileRandomStrategy,
    "entropy_pfilter_entropy": EntropyPercentileEntropyStrategy,
    "bald_pfilter_random": BALDPercentileRandomStrategy,
    "bald_pfilter_bald": BALDPercentileBALDStrategy,
    "coreset": CoreSetStrategy,
    "kcenter": CoreSetStrategy,
    "core_set": CoreSetStrategy,
    "bald_dual_a": BALDDualAStrategy,
    "bald_dual_b": BALDDualBStrategy,
    "bald_adv_lagrangian": BALDAdvLagrangianStrategy,
    "ours": OursStrategy,
    "ours_hessian": OursHessianStrategy,
    "ours_gap": OursGapStrategy,
    "ours_grad_disp": OursGradDispStrategy,
}


def build_acquisition_strategy(name: str, cfg):
    name = name.lower()
    m = re.fullmatch(
        r"((?:logdet_adv_disp|semantic_logdet|adv_displacement_logdet)(?:_swap)?|logit_adv_disp_swap|logdet_adv_feat_swap|logdet_adv_logit_swap|logit_adv_feat_swap|logit_adv_logit_swap)_p(\d+)",
        name,
    )
    if m is not None:
        pct = int(m.group(2))
        if pct < 0 or pct > 100:
            raise ValueError(f"Unsupported acquisition method percentile suffix: {name}")
        if hasattr(cfg, "logdet_adv_disp_percentile"):
            cfg.logdet_adv_disp_percentile = float(pct) / 100.0
        name = m.group(1)
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unsupported acquisition method: {name}")
    return METHOD_REGISTRY[name](cfg)
