from acquisition.bait import BAITStrategy
from acquisition.badge import BADGEStrategy
from acquisition.badge_dual_a import BADGEDualAStrategy
from acquisition.badge_dual_b import BADGEDualBStrategy
from acquisition.badge_adv_lagrangian import BADGEAdvLagrangianStrategy
from acquisition.badge_adv_mult import BADGEAdvMultStrategy
from acquisition.bald_adv_lagrangian import BALDAdvLagrangianStrategy
from acquisition.bald import BALDStrategy
from acquisition.bald_dual_a import BALDDualAStrategy
from acquisition.bald_dual_b import BALDDualBStrategy
from acquisition.entropy import EntropyStrategy
from acquisition.entropy_dual_a import EntropyDualAStrategy
from acquisition.entropy_dual_b import EntropyDualBStrategy
from acquisition.saal import SAALStrategy
from acquisition.saal_dual_b import SAALDualBStrategy
from acquisition.margin import MarginStrategy
from acquisition.margin_dual_b import MarginDualBStrategy
from acquisition.ours import OursGapStrategy, OursGradDispStrategy, OursHessianStrategy, OursStrategy
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
    "badge": BADGEStrategy,
    "badge_dual_a": BADGEDualAStrategy,
    "badge_dual_b": BADGEDualBStrategy,
    "badge_adv_mult": BADGEAdvMultStrategy,
    "badge_adv_lagrangian": BADGEAdvLagrangianStrategy,
    "bait": BAITStrategy,
    "bald": BALDStrategy,
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
    if name not in METHOD_REGISTRY:
        raise ValueError(f"Unsupported acquisition method: {name}")
    return METHOD_REGISTRY[name](cfg)
