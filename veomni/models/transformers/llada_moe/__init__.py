from ...loader import MODEL_CONFIG_REGISTRY, MODELING_REGISTRY


@MODEL_CONFIG_REGISTRY.register("llada_moe")
def register_llada_moe_config():
    from .configuration_llada_moe import LLaDAConfig

    return LLaDAConfig


@MODELING_REGISTRY.register("llada_moe")
def register_llada_moe_modeling(architecture: str):
    from .modeling_llada_moe import (
        LLaDAMoEModelLM,
        LLaDAMoEModel,
    )

    if architecture == "LLaDAMoEModelLM":
        return LLaDAMoEModelLM
    elif architecture == "LLaDAMoEModel":
        return LLaDAMoEModel
    else:
        raise NotImplementedError(architecture)
