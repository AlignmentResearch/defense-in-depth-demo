# Inspired by https://gist.github.com/s-mawjee/ad0d8e0c7e07265cae097899fe48c023


import nvidia_smi

from robust_llm import logger


def _check_gpu():
    nvidia_smi.nvmlInit()
    num_gpu = nvidia_smi.nvmlDeviceGetCount()
    gpu = num_gpu > 0

    return gpu, num_gpu


def _bytes_to_megabytes(bytes):
    return round((bytes / 1024) / 1024, 2)


def print_gpu_usage():
    gpu, num_gpu = _check_gpu()
    if not gpu:
        logger.info("No GPU found, so no usage to report.")
        return

    for i in range(num_gpu):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        logger.info(
            "GPU-%s: GPU-Memory: %s/%s MB",
            i,
            _bytes_to_megabytes(info.used),
            _bytes_to_megabytes(info.total),
        )


def assert_dicts_equal(dict1: dict, dict2: dict) -> bool:
    if dict1 == dict2:
        return True

    differences = []
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())

    # Check for missing or extra keys
    missing_keys = keys1 - keys2
    extra_keys = keys2 - keys1

    if missing_keys:
        differences.append(f"Missing keys in second dict: {missing_keys}")
    if extra_keys:
        differences.append(f"Extra keys in second dict: {extra_keys}")

    # Check for value differences in common keys
    common_keys = keys1 & keys2
    for key in common_keys:
        if dict1[key] != dict2[key]:
            differences.append(
                f"Different values for key '{key}': {dict1[key]} != {dict2[key]}"
            )

    if differences:
        raise AssertionError("Dictionaries are not equal:\n" + "\n".join(differences))

    return True
