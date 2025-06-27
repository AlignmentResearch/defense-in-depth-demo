METRICS = [
    "adversarial_eval/attack_success_rate",
]

SUMMARY_KEYS = [
    "experiment_yaml.model.name_or_path",
    "experiment_yaml.training.seed",
    "experiment_yaml.training.adversarial.num_examples_to_generate_each_round",
    "experiment_yaml.dataset.n_train",
    "model_size",
]

PROJECT_NAME = "farai/robust-llm"

MODEL_NAMES = {
    "pythia": [
        "14m",
        "31m",
        "70m",
        "160m",
        "410m",
        "1b",
        "1.4b",
        "2.8b",
        "6.9b",
        "12b",
    ],
    "qwen": ["0.5B", "1.5B", "3B", "7B", "14B"],
}


MODEL_SIZES = {
    "clf": {
        "pythia": [
            7629056,
            17617408,
            44672000,
            123691008,
            353824768,
            908763136,  # 1b
            1311629312,  # 1.4b
            2646435840,  # 2.8b
            6650740736,  # 6.9b
            11586560000,  # 12b
        ],
        "qwen": [494034560, 1543717376, 3085942784, 7070626304, 13991476224],
    },
    "gen": {
        "pythia": [
            7628800,
            17616896,
            44670976,
            123689472,
            353822720,
            908759040,
            1311625216,
            2646430720,
            6650732544,
            11586549760,
        ],
        "qwen": [494032768, 1543714304, 3085938688, 7615616512, 14770033664],
    },
}


def estimate_flops_for_training(n_params: int, n_training_tokens: int) -> int:
    """Estimate training flops using 6ND.

    This estimate is commonly used and appears in
    https://arxiv.org/pdf/2001.08361#page=9.71
    """
    return 6 * n_params * n_training_tokens


# Training tokens from https://github.com/EleutherAI/pythia and
# https://arxiv.org/pdf/2412.15115
n_training_tokens = {
    "pythia": dict.fromkeys(MODEL_NAMES["pythia"], 299_892_736_000),
    "qwen": dict.fromkeys(MODEL_NAMES["qwen"], 18_000_000_000_000),
}

ESTIMATED_PRETRAIN_COMPUTE = {
    family: {
        model_idx: estimate_flops_for_training(
            n_params, tokens_dict[MODEL_NAMES[family][model_idx]]
        )
        for model_idx, n_params in enumerate(MODEL_SIZES["gen"][family])
    }
    for family, tokens_dict in n_training_tokens.items()
}
