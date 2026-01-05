from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("llada_emoe")
def register_llada_moe_config():
    from .configuration_llada_emoe import LLaDAConfig

    return LLaDAConfig


@MODELING_REGISTRY.register("llada_emoe")
def register_llada_moe_modeling(architecture: str):
    from .modeling_llada_emoe import (
        LLaDAEMoEModelLM,
        LLaDAEMoEModel,
    )

    if architecture == "LLaDAEMoEModelLM":
        return LLaDAEMoEModelLM
    elif architecture == "LLaDAEMoEModel":
        return LLaDAEMoEModel
    else:
        raise NotImplementedError(architecture)
