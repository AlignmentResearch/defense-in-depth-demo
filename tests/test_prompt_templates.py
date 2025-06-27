import pytest

from robust_llm.dist_utils import DistributedRNG
from robust_llm.models.prompt_templates import AttackChunks


def test_chunks_to_prompt_template_prefix():
    rng = DistributedRNG(0, None)
    chunks = AttackChunks(
        unmodifiable_prefix="unmodifiable_prefix",
        modifiable_infix="modifiable_infix",
        unmodifiable_suffix="unmodifiable_suffix",
    )
    prompt_template = chunks.get_prompt_template(0, 0, rng)
    assert prompt_template.before_attack == "unmodifiable_prefix"
    assert (
        prompt_template.after_attack
        == "!ATTACK_STRING_PLACEHOLDER!modifiable_infixunmodifiable_suffix"
    )
    assert (
        prompt_template.build_prompt(attack_text="attack_text")
        == "unmodifiable_prefixattack_textmodifiable_infixunmodifiable_suffix"
    )


def test_chunks_to_prompt_template_suffix():
    rng = DistributedRNG(0, None)
    chunks = AttackChunks(
        unmodifiable_prefix="unmodifiable_prefix",
        modifiable_infix="modifiable_infix",
        unmodifiable_suffix="unmodifiable_suffix",
    )
    prompt_template = chunks.get_prompt_template(1, 1, rng)
    assert prompt_template.before_attack == "unmodifiable_prefixmodifiable_infix"
    assert (
        prompt_template.after_attack == "!ATTACK_STRING_PLACEHOLDER!unmodifiable_suffix"
    )
    assert (
        prompt_template.build_prompt(attack_text="attack_text")
        == "unmodifiable_prefixmodifiable_infixattack_textunmodifiable_suffix"
    )


def test_chunks_to_prompt_template_suffix_75_percent():
    rng = DistributedRNG(0, None)
    chunks = AttackChunks(
        unmodifiable_prefix="unmodifiable_prefix",
        modifiable_infix="modifiable_infix",
        unmodifiable_suffix="unmodifiable_suffix",
    )
    prompt_template = chunks.get_prompt_template(0.75, 0.75, rng)
    assert prompt_template.before_attack == "unmodifiable_prefixmodifiable_i"
    assert (
        prompt_template.after_attack
        == "!ATTACK_STRING_PLACEHOLDER!nfixunmodifiable_suffix"
    )
    assert (
        prompt_template.build_prompt(attack_text="attack_text")
        == "unmodifiable_prefixmodifiable_iattack_textnfixunmodifiable_suffix"
    )


def test_chunks_to_prompt_template_placeholder():
    rng = DistributedRNG(0, None)
    chunks = AttackChunks(
        unmodifiable_prefix="unmodifiable_prefix",
        modifiable_infix="modifiable_infix!ATTACK_STRING_PLACEHOLDER!",
        unmodifiable_suffix="unmodifiable_suffix",
    )
    prompt_template = chunks.get_prompt_template(None, None, rng)
    assert prompt_template.before_attack == "unmodifiable_prefixmodifiable_infix"
    assert (
        prompt_template.after_attack == "!ATTACK_STRING_PLACEHOLDER!unmodifiable_suffix"
    )
    assert (
        prompt_template.build_prompt(attack_text="attack_text")
        == "unmodifiable_prefixmodifiable_infixattack_textunmodifiable_suffix"
    )


@pytest.mark.parametrize(
    ["perturb_min", "perturb_max"], [(0.5, 0.4), (-0.1, 0.0), (0.0, 1.1)]
)
def test_chunks_to_prompt_template_raises(perturb_min: float, perturb_max: float):
    rng = DistributedRNG(0, None)
    chunks = AttackChunks(
        unmodifiable_prefix="unmodifiable_prefix",
        modifiable_infix="modifiable_infix",
        unmodifiable_suffix="unmodifiable_suffix",
    )
    with pytest.raises(AssertionError):
        chunks.get_prompt_template(perturb_min, perturb_max, rng)
