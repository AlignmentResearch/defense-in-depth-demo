import pytest
from datasets import ClassLabel
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.config.dataset_configs import MessageFilter
from robust_llm.dist_utils import DistributedRNG
from robust_llm.rllm_datasets.load_rllm_dataset import load_rllm_dataset
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset


@pytest.fixture()
def clf_dataset() -> RLLMDataset:
    """Fixture for the PasswordMatch dataset.

    We use the 'pos' version of the dataset with all positive examples so we know
    in advance that the `clf_label`s should be 1 and we can easily flip them by
    inserting any other word.
    """

    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        revision="2.1.0",
        n_train=5,
        n_val=5,
        config_name="pos",
        inference_type="classification",
        strip_leading_whitespace=False,
    )
    dataset = load_rllm_dataset(cfg, split="validation")
    return dataset


@pytest.fixture()
def gen_dataset() -> RLLMDataset:
    """Fixture for the PasswordMatch dataset.

    We use the 'pos' version of the dataset with all positive examples so we know
    in advance that the `clf_label`s should be 1 and we can easily flip them by
    inserting any other word.
    """

    cfg = DatasetConfig(
        dataset_type="AlignmentResearch/PasswordMatch",
        revision="2.1.0",
        n_train=5,
        n_val=5,
        config_name="pos",
        inference_type="generation",
        strip_leading_whitespace=True,
    )
    dataset = load_rllm_dataset(cfg, split="validation")
    return dataset


@pytest.fixture()
def tokenizer():
    return AutoTokenizer.from_pretrained("gpt2")


def test_initialized_rllm_dataset(clf_dataset: RLLMDataset):
    assert clf_dataset.num_classes == 2
    # Length should match underlying dataset
    assert len(clf_dataset) == len(clf_dataset.ds)
    assert len(clf_dataset) == 5
    # We didn't tokenize yet
    assert not clf_dataset.is_tokenized
    # 'clf_label' should be a ClassLabel feature
    assert isinstance(clf_dataset.ds.features["clf_label"], ClassLabel)


def test_tokenization_and_subset(clf_dataset: RLLMDataset, tokenizer):
    tokenized_dataset = clf_dataset.tokenize(
        tokenizer,
        message_filter=MessageFilter.INPUT,
        system_prompt=None,
    )
    # We should have tokenized the dataset
    assert tokenized_dataset.is_tokenized

    # Check that the tokenized dataset has the right keys
    assert {"input_ids", "attention_mask"} <= set(tokenized_dataset.ds[0].keys())
    smaller_dataset = tokenized_dataset.get_random_subset(2, seed=42, accelerator=None)
    assert len(smaller_dataset) == 2
    assert smaller_dataset.is_tokenized

    smallest_dataset = tokenized_dataset.get_random_subset(
        1, generator=DistributedRNG(42, accelerator=None)
    )
    assert len(smallest_dataset) == 1
    assert smallest_dataset.is_tokenized

    with pytest.raises(AssertionError):
        tokenized_dataset.get_random_subset(
            2, seed=42, generator=DistributedRNG(42, accelerator=None)
        )


def test_with_attacked_text(clf_dataset: RLLMDataset, tokenizer):
    attacked_texts = []
    for example in clf_dataset.ds:
        assert isinstance(example, dict)
        assert example["clf_label"] == 1
        chunks = example["chunked_text"][:]
        chunks[2] = "some_other_word"
        attacked_text = "".join(chunks)
        attacked_texts.append(attacked_text)

    attacked_dataset = clf_dataset.with_attacked_text(attacked_texts)
    assert len(attacked_dataset) == len(clf_dataset)
    # Check that the attacked dataset has the original text in it
    assert attacked_dataset.ds["text"] == clf_dataset.ds["text"]
    # Check that the labels are not flipped (they previously were flipped for
    # old PasswordMatch, so we make sure this is not longer the case.)
    assert all(
        [ex["attacked_clf_label"] == 1 for ex in attacked_dataset.ds]  # type: ignore
    )
    assert all([ex["clf_label"] == 1 for ex in attacked_dataset.ds])  # type: ignore
    # Check that 'attacked_clf_label' is still a ClassLabel feature
    assert isinstance(attacked_dataset.ds.features["attacked_clf_label"], ClassLabel)

    # Test 'as_adversarial_examples'
    with pytest.raises(ValueError):
        # Dataset has to be tokenized first
        adv_dataset = attacked_dataset.as_adversarial_examples(system_prompt=None)

    adv_dataset = attacked_dataset.tokenize(
        tokenizer, message_filter=MessageFilter.INPUT, system_prompt=None
    ).as_adversarial_examples(system_prompt=None)
    assert len(adv_dataset) == len(clf_dataset)
    assert adv_dataset.ds["text"] == attacked_dataset.ds["attacked_text"]
    assert all([ex["clf_label"] == 1 for ex in adv_dataset.ds])  # type: ignore
    # Check that the 'attacked_text' and 'attacked_clf_label' columns are gone
    assert "attacked_text" not in adv_dataset.ds
    assert "attacked_clf_label" not in adv_dataset.ds
    # Check that the new 'clf_label' is still a ClassLabel feature
    assert isinstance(adv_dataset.ds.features["clf_label"], ClassLabel)


def test_for_training_clf(clf_dataset: RLLMDataset, tokenizer):
    with pytest.raises(AssertionError):
        clf_dataset.for_training()

    tokenized_dataset = clf_dataset.tokenize(
        tokenizer, message_filter=MessageFilter.INPUT, system_prompt=None
    )
    hf_ds = tokenized_dataset.for_training()
    assert len(hf_ds) == len(tokenized_dataset)
    # Check that the features are correct
    assert "input_ids" in hf_ds.features
    assert "attention_mask" in hf_ds.features
    assert "label" in hf_ds.features
    assert "clf_label" not in hf_ds.features
    assert "chunked_text" not in hf_ds.features
    assert isinstance(hf_ds.features["label"], ClassLabel)


def test_for_training_gen(gen_dataset: RLLMDataset, tokenizer):
    with pytest.raises(AssertionError):
        gen_dataset.for_training()

    tokenized_dataset = gen_dataset.tokenize(
        tokenizer, message_filter=MessageFilter.INPUT, system_prompt=None
    )
    hf_ds = tokenized_dataset.for_training()
    assert len(hf_ds) == len(tokenized_dataset)
    # Check that the features are correct
    assert "input_ids" in hf_ds.features
    assert "attention_mask" in hf_ds.features
    assert "label" not in hf_ds.features
    assert "clf_label" not in hf_ds.features
    assert "chunked_text" not in hf_ds.features

    for hf_example, rllm_example in zip(hf_ds, tokenized_dataset.ds):
        assert isinstance(hf_example, dict) and isinstance(rllm_example, dict)
        full_text = rllm_example["text"] + rllm_example["gen_target"]
        decoded = tokenizer.decode(hf_example["input_ids"], skip_special_tokens=True)
        assert decoded == full_text
        # Quick test that this equality is meaningful.
        assert decoded != 2 * full_text
