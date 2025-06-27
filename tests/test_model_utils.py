import numpy as np
import torch
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from robust_llm.models.model_output_metrics import GenerationMetric, Metric
from robust_llm.models.model_utils import MASK_LABEL, lm_loss, mean_of_batch_losses


def test_generation_successes_from_logits():
    input_ids = torch.tensor(
        [
            # 100 is first token (which is not predicted), 101 is pad
            [100, 0, 1, 0, 1, 101],
            [100, 0, 1, 0, 1, 101],
            [100, 0, 1, 0, 1, 101],
        ]
    ).tolist()

    logits = torch.tensor(
        [
            # Argmax of the first example is [1, 1, 0, 1, 1, ?]
            # Zeroth logit is based on input_ids[0]=100 and predicts input_ids[1]=0
            # First logit is based on input_ids[1]=0 and predicts input_ids[2]=1
            # Second logit is based on input_ids[2]=1 and predicts input_ids[3]=0
            # Third logit is based on input_ids[3]=0 and predicts input_ids[4]=1
            # Fourth logit is based on input_ids[4]=1 and predicts input_ids[5]=101
            [
                [0.1, 0.9],
                [0.1, 0.9],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.0, 0.0],
            ],
            # Argmax of the second example is [1, 0, 0, 1, 1, ?]
            [
                [0.1, 0.9],
                [0.9, 0.1],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.0, 0.0],
            ],
            # Argmax of the third example is [1, 0, 1, 1, 0, ?] (based on data)
            [
                [0.1, 0.9],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.9, 0.1],
                [0.0, 0.0],
            ],
        ]
    )
    goal = [[1, 0, 1], [1, 0, 1], [1, 0, 1]]

    out = GenerationMetric.from_logits(
        logits=logits, input_ids=input_ids, goal=goal, metric=Metric.SUCCESS
    )
    # Predictions made FROM the goal tokens take up indices 1, 2, 3.
    # Therefore predictions OF the goal tokens (input_ids[2:5]) are the argmax
    # of logprobs at indices 1, 2, 3.
    # Example 1: Predictions at [1,2,3] are [1, 0, 1]. Matches goal. Success.
    # Example 2: Predictions at [1,2,3] are [0, 0, 1]. Doesn't match goal. Fail.
    # Example 3: Predictions at [1,2,3] are [0, 1, 1]. Doesn't match goal. Fail.
    assert out == [True, False, False]


def test_generation_successes_from_logits_with_caching():
    """With caching we get fewer logits back."""
    input_ids = torch.tensor(
        [
            # 100 is first token (which is not predicted), 101 is pad
            [100, 0, 1, 0, 1, 101],
            [100, 0, 1, 0, 1, 101],
        ]
    ).tolist()

    logits = torch.tensor(
        [
            # Second logit is based on input_ids[2]=1 and predicts input_ids[3]=0
            # Third logit is based on input_ids[3]=0 and predicts input_ids[4]=1
            # Fourth logit is based on input_ids[4]=1 and predicts input_ids[5]=101
            [
                [0.1, 0.9],
                [0.1, 0.9],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.0, 0.0],
            ],
            # Argmax of the second example is [1, 0, 1, 1, ?, ?] (based on data)
            # Convert list of dicts to a tensor [seq_len, vocab_size]
            [
                [0.1, 0.9],
                [0.9, 0.1],
                [0.1, 0.9],
                [0.1, 0.9],
                [0.0, 0.0],
                [0.0, 0.0],
            ],
        ]
    )
    goal = [[1, 0, 1], [1, 0, 1]]

    out = GenerationMetric.from_logits(
        logits=logits, input_ids=input_ids, goal=goal, metric=Metric.SUCCESS
    )
    # Predictions made FROM the goal tokens take up indices 1, 2, 3.
    # Therefore predictions OF the goal tokens (input_ids[2:5]) are the argmax
    # of logprobs at indices 1, 2, 3.
    # Example 1: Predictions at [1,2,3] are [1, 0, 1]. Matches goal. Success.
    # Example 2: Predictions at [1,2,3] are [0, 1, 1]. Doesn't match goal. Fail.
    assert out == [True, False]


def test_lm_loss_handwritten(p: float = 0.9):
    logits = torch.tensor(
        [
            [
                [np.log(1 - p), np.log(p)],
                [np.log(p), np.log(1 - p)],
                [np.log(1 - p), np.log(p)],
                [np.log(p), np.log(1 - p)],
                [np.log(1 - p), np.log(p)],
                [0, 0],
            ],
            [
                [np.log(p), np.log(1 - p)],
                [np.log(1 - p), np.log(p)],
                [np.log(p), np.log(1 - p)],
                [np.log(1 - p), np.log(p)],
                [0, 0],
                [0, 0],
            ],
        ]
    )
    labels = torch.tensor(
        [
            [0, 1, 0, 1, 0, 1],
            [1, 0, 1, 0, MASK_LABEL, MASK_LABEL],
        ]
    )
    loss = lm_loss(logits, labels)
    assert all(np.isclose(loss.tolist(), [-np.log(p)] * logits.shape[0]))


def test_lm_loss_vs_hf():
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    batch = tokenizer(
        ["Hello, world!", "Greetings, fellow humans!"],
        return_tensors="pt",
        padding=True,
    )
    batch["labels"] = batch["input_ids"].masked_fill(  # type: ignore
        batch["attention_mask"] == 0, MASK_LABEL
    )
    model_output = model(**batch)
    hf_loss = model_output.loss
    our_loss = lm_loss(model_output.logits, batch["labels"])  # type: ignore
    mean_from_our_loss = mean_of_batch_losses(
        our_loss,
        batch["attention_mask"],  # type: ignore
    )
    assert np.isclose(hf_loss.item(), mean_from_our_loss)
