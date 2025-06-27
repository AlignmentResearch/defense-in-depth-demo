from robust_llm.config.dataset_configs import DatasetConfig
from robust_llm.rllm_datasets.rllm_dataset import RLLMDataset
from robust_llm.utils import print_time


@print_time()
def load_rllm_dataset(dataset_config: DatasetConfig, split: str) -> RLLMDataset:
    """Loads a dataset from huggingface based on its name."""
    dataset_type = dataset_config.dataset_type
    dataset_class = RLLMDataset.get_dataset(dataset_type)
    return dataset_class(dataset_config, split)
