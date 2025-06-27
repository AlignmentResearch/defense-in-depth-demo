import datetime
from unittest.mock import patch

from robust_llm.models.model_disk_utils import (
    generate_model_save_path,
    get_model_load_path,
    mark_model_save_as_finished,
)


def test_get_model_load_path(tmp_path):
    """Tests basic usage of model_load_path().

    Also indirectly tests generate_model_save_path() and
    mark_model_save_as_finished().
    """
    models_path = tmp_path
    model_name = "test-model"
    revision = "test-revision"

    # Loading a non-existent model should fail.
    assert (
        get_model_load_path(
            models_path=models_path, model_name=model_name, revision=revision
        )
        is None
    )

    model_versions = []
    for i in range(2):
        with patch("datetime.datetime", wraps=datetime.datetime) as mock_datetime:
            mock_datetime.now.return_value = datetime.datetime(2024, 1, 1, i, 0, 0)
            model_versions.append(
                generate_model_save_path(
                    models_path=models_path, model_name=model_name, revision=revision
                )
            )

    for path in model_versions:
        path.mkdir(parents=True)

    # None of the model versions are done saving, so get_model_load_path should
    # still fail.
    assert (
        get_model_load_path(
            models_path=models_path, model_name=model_name, revision=revision
        )
        is None
    )

    # After a model finishes saving, it should be returned by
    # get_model_load_path().
    mark_model_save_as_finished(model_save_directory=model_versions[0])
    assert (
        get_model_load_path(
            models_path=models_path, model_name=model_name, revision=revision
        )
        == model_versions[0]
    )

    # If there are multiple models saved, the most recent one should be
    # returned.
    mark_model_save_as_finished(model_save_directory=model_versions[1])
    assert (
        get_model_load_path(
            models_path=models_path, model_name=model_name, revision=revision
        )
        == model_versions[1]
    )
