from typing import Optional

from accelerate import Accelerator

from robust_llm.config.layered_defense_configs import DefenseComponentConfig
from robust_llm.layered_defense.components import (
    ComponentRegistry,
    DefenseComponent,
    InputComponent,
    OutputComponent,
)
from robust_llm.models.wrapped_model import WrappedModel


def create_defense_component(
    config: DefenseComponentConfig,
    victim: Optional[WrappedModel],
    accelerator: Accelerator,
) -> DefenseComponent:
    component_cls = ComponentRegistry.get(config.name)
    # Get all fields from the config that aren't in the base DefenseComponentConfig
    extra_kwargs = {
        k: v
        for k, v in vars(config).items()
        if k not in vars(DefenseComponentConfig())
        and not k.startswith("_")  # Skip internal attrs
    }
    # Combine the extra kwargs with the standard kwargs
    all_kwargs = {**config.kwargs, **extra_kwargs}
    return component_cls(victim, accelerator, **all_kwargs)


def create_input_component(
    config: DefenseComponentConfig,
    victim: Optional[WrappedModel],
    accelerator: Accelerator,
) -> InputComponent:
    component = create_defense_component(config, victim, accelerator)
    assert isinstance(component, InputComponent)
    return component


def create_output_component(
    config: DefenseComponentConfig,
    victim: Optional[WrappedModel],
    accelerator: Accelerator,
) -> OutputComponent:
    component = create_defense_component(config, victim, accelerator)
    assert isinstance(component, OutputComponent)
    return component
