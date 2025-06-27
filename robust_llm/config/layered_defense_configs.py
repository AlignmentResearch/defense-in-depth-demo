from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from robust_llm.config.constants import ProbeType
from robust_llm.config.model_configs import ModelConfig



@dataclass
class DefenseComponentConfig:
    """Config used in defense component setup."""

    name: str = MISSING
    kwargs: dict = field(default_factory=dict)


@dataclass
class ThresholdComponentConfig(DefenseComponentConfig):
    """Base for components that have a threshold."""

    threshold: float = MISSING


@dataclass
class FilterComponentConfig(ThresholdComponentConfig):
    """Config used in filter defense component setup."""

    filter_config: ModelConfig = MISSING


@dataclass
class PromptedFilterTemplateConfig:
    """Template to use for querying the prompted filter."""

    template: str = MISSING

    def __post_init__(self):
        if "{content_id}" not in self.template or "{content}" not in self.template:
            raise ValueError("template must contain {content_id} and {content}")


@dataclass
class PromptedFilterComponentConfig(FilterComponentConfig):
    """Config used in prompted filter defense component setup."""

    template_config: PromptedFilterTemplateConfig = MISSING


@dataclass
class ProbeConfig:
    """Config used in probe setup."""

    model_name: str = MISSING
    revision: str = "main"
    probe_type: ProbeType = ProbeType.LINEAR
    layer: int = MISSING
    num_heads: int | None = None


@dataclass
class ProbeComponentConfig(ThresholdComponentConfig):
    """Config used in probe defense component setup."""

    probe_config: ProbeConfig = MISSING


@dataclass
class LayeredDefenseConfig:
    """
    Configs used in layered defense setup.

    For now, we get the victim model from the main ModelConfig.
    """

    input_components: dict[str, DefenseComponentConfig] = field(default_factory=dict)
    output_components: dict[str, DefenseComponentConfig] = field(default_factory=dict)


cs = ConfigStore.instance()
cs.store(name="DEFAULT", group="layered_defense", node=LayeredDefenseConfig)
cs.store(
    name="PROMPT_GUARD",
    group="layered_defense/components",
    node=DefenseComponentConfig(
        name="PromptGuardInputFilter",
        kwargs={"model_name": "meta-llama/Prompt-Guard-86M"},
    ),
)
cs.store(
    name="LLAMA_GUARD",
    group="layered_defense/components",
    node=ThresholdComponentConfig(
        name="LlamaGuard3InputFilter",
        kwargs={"model_name": "meta-llama/Llama-Guard-3-8B"},
    ),
)
cs.store(
    name="TRAINED_INPUT_PROBE",
    group="layered_defense/components",
    node=ProbeComponentConfig(
        name="TrainedInputProbe",
    ),
)
cs.store(
    name="TRAINED_OUTPUT_PROBE",
    group="layered_defense/components",
    node=ProbeComponentConfig(
        name="TrainedOutputProbe",
    ),
)

cs.store(
    name="TRAINED_INPUT_FILTER",
    group="layered_defense/components",
    node=FilterComponentConfig(
        name="TrainedInputFilter",
    ),
)
cs.store(
    name="TRAINED_OUTPUT_FILTER",
    group="layered_defense/components",
    node=FilterComponentConfig(
        name="TrainedOutputFilter",
    ),
)
cs.store(
    name="TRAINED_TRANSCRIPT_FILTER",
    group="layered_defense/components",
    node=FilterComponentConfig(
        name="TrainedTranscriptFilter",
    ),
)
cs.store(
    name="PROMPTED_INPUT_FILTER",
    group="layered_defense/components",
    node=PromptedFilterComponentConfig(
        name="PromptedInputFilter",
    ),
)
cs.store(
    name="PROMPTED_OUTPUT_FILTER",
    group="layered_defense/components",
    node=PromptedFilterComponentConfig(
        name="PromptedOutputFilter",
    ),
)
# Thresholds are from
# robust_llm.plotting_utils.pipeline_analysis.FILTER_THRESHOLDS.
cs.store(
    name="WILDGUARD_INPUT_FILTER",
    group="layered_defense/components/input_filter",
    node=ThresholdComponentConfig(
        name="WildGuardInputFilter",
        threshold=0.012920428067445755,
    ),
)
cs.store(
    name="WILDGUARD_OUTPUT_FILTER",
    group="layered_defense/components/output_filter",
    node=ThresholdComponentConfig(
        name="WildGuardOutputFilter",
        threshold=0.0005072808708064258,
    ),
)
cs.store(
    name="SHIELDGEMMA_INPUT_FILTER",
    group="layered_defense/components",
    node=ThresholdComponentConfig(
        name="ShieldGemmaInputFilter",
        threshold=0.0103302001953125,
    ),
)
cs.store(
    name="SHIELDGEMMA_OUTPUT_FILTER",
    group="layered_defense/components",
    node=ThresholdComponentConfig(
        name="ShieldGemmaOutputFilter",
        threshold=0.005220125894993544,
    ),
)
