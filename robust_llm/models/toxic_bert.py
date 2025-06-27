from accelerate import Accelerator
from detoxify import Detoxify
from transformers.models.bert.modeling_bert import BertForSequenceClassification
from transformers.models.bert.tokenization_bert import BertTokenizer

MODEL_NAME = "unitary/toxic-bert"


class ToxicBert(Detoxify):
    """Wrapper around the Detoxify model for FSDP compatibility.

    The original Detoxify __init__ method tries to load a checkpoint rather
    than a model by name, which breaks with FSDP. This class is a workaround
    to load the model by name instead when setting up the model.
    Issue: https://github.com/pytorch/pytorch/issues/130875
    """

    def __init__(
        self,
        accelerator: Accelerator,
    ) -> None:
        self.model = BertForSequenceClassification.from_pretrained(
            pretrained_model_name_or_path=MODEL_NAME
        )
        assert isinstance(self.model, BertForSequenceClassification)
        self.model = self.model.to(device=accelerator.device)  # type: ignore
        self.tokenizer = BertTokenizer.from_pretrained(
            MODEL_NAME, device=accelerator.device
        )
        self.class_names = [
            "toxicity",
            "severe_toxicity",
            "obscene",
            "threat",
            "insult",
            "identity_attack",
        ]
