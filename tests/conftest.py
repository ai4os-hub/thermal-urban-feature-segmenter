"""Generic tests environment configuration. This file implement all generic
fixtures to simplify model and api specific testing.

Modify this file only if you need to add new fixtures or modify the existing
related to the environment and generic tests.
"""
# pylint: disable=redefined-outer-name
from datetime import datetime
import inspect
import json
import os
import pathlib
import shutil
import tempfile

from unittest.mock import patch
import pytest

import api


@pytest.fixture(scope="session", autouse=True)
def tests_datapath():
    """Fixture to generate a original directory path for datasets."""
    return pathlib.Path(api.config.DATA_PATH).absolute()


@pytest.fixture(scope="session", autouse=True)
def tests_modelspath():
    """Fixture to generate a original directory path for models."""
    return pathlib.Path(api.config.MODELS_PATH).absolute()


@pytest.fixture(scope="session", params=os.listdir("tests/configurations"))
def config_file(request):
    """Fixture to provide each deepaas configuration path."""
    config_str = f"tests/configurations/{request.param}"
    return pathlib.Path(config_str).absolute()


@pytest.fixture(scope="module", name="tmptestsdir")
def create_tmptestsdir():
    """Fixture to generate a temporary directory for each test module."""
    with tempfile.TemporaryDirectory() as tmptestsdir:
        os.chdir(tmptestsdir)
        yield tmptestsdir


@pytest.fixture(scope="module", autouse=True)
def copytree_data(tmptestsdir, tests_datapath):
    """Fixture to copy the original data directory to the test directory."""
    shutil.copytree(tests_datapath, f"{tmptestsdir}/{api.config.DATA_PATH}")


@pytest.fixture(scope="module", autouse=True)
def copytree_models(tmptestsdir, tests_modelspath):
    """Fixture to copy the original models directory to the test directory."""
    shutil.copytree(tests_modelspath,
                    f"{tmptestsdir}/{api.config.MODELS_PATH}")


def generate_signature(names, kind=inspect.Parameter.POSITIONAL_OR_KEYWORD):
    """Function to generate signatures dynamically."""
    parameters = [inspect.Parameter(name, kind) for name in names]
    return inspect.Signature(parameters=parameters)


def generate_fields_fixture(signature):
    """Function to generate fixtures dynamically with dynamic arguments."""
    def fixture_function(**options):  # fmt: skip
        return {k: v for k, v in options.items()}
        # with if v = None to include inputs that are None
    fixture_function.__signature__ = signature
    return pytest.fixture(scope="module")(fixture_function)


@pytest.fixture(scope="module")
def patch_get_remote_dirs():
    """Patch to replace get_remote_dirs (NextCloud access)"""

    with patch(
        "api.utils.get_remote_dirs", autospec=True
    ) as mock_get_remote_dirs:
        mock_get_remote_dirs.return_value = [
            "dummy/folder1/", "dummy/folder2/"
        ]
        yield mock_get_remote_dirs


@pytest.fixture(scope="module")
def metadata(patch_get_remote_dirs):
    """Fixture to return get_metadata to assert properties."""
    return api.get_metadata()


# Generate and inject fixtures for predict arguments
fields_predict = api.schemas.PredArgsSchema().fields
signature = generate_signature(fields_predict.keys())
globals()["predict_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def predictions(patch_get_remote_dirs, predict_kwds):
    """Fixture to return predictions to assert properties."""
    return api.predict(**predict_kwds)


# Generate and inject fixtures for training arguments
fields_training = api.schemas.TrainArgsSchema().fields
signature = generate_signature(fields_training.keys())
globals()["training_kwds"] = generate_fields_fixture(signature)


@pytest.fixture(scope="module")
def training(patch_run_bash_subprocess, training_kwds):
    """Fixture to return training to assert properties."""
    return api.train(**training_kwds)


@pytest.fixture(scope="module")
def patch_run_bash_subprocess(tmptestsdir):
    """Patch to replace run_bash_subprocess (train.sh execution)"""

    with patch(
        "tufsegm_api.utils.run_bash_subprocess", autospec=True
    ) as mock_run_bash_subprocess:
        # create dummy training folder and content
        mock_model_path = pathlib.Path(
            tmptestsdir,
            api.config.MODELS_PATH,
            datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        )
        mock_model_path.mkdir(exist_ok=True)

        with open(
            pathlib.Path(mock_model_path, "eval.json"), "w"
        ) as mock_json_file:
            json.dump({"mock metrics": 0.0}, mock_json_file)
        # run training with a mock subprocess
        mock_run_bash_subprocess.return_value = 0
        yield mock_run_bash_subprocess
