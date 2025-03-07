"""Endpoint functions to integrate the submodule TUFSeg with the DEEPaaS API.

For more information about how to edit the module see, take a look at the
docs [1] and at a canonical exemplar module [2].

[1]: https://docs.ai4eosc.eu/
[2]: https://github.com/deephdc/demo_app
"""
import getpass
import logging
import os
from pathlib import Path
import shutil

from aiohttp.web import HTTPException

import tufsegm_api as aimodel

from . import config, responses, schemas, utils

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_metadata():
    """Returns a dictionary containing metadata information about the module.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        A dictionary containing metadata information required by DEEPaaS.
    """
    try:
        logger.info("GET 'metadata' called. Collected data from: %s",
                    config.API_NAME)
        model_name = config.MODEL_TYPE + config.MODEL_SUFFIX
        metadata = {
            "author": config.API_METADATA.get("authors"),
            "author-email": config.API_METADATA.get("author-emails"),
            "description": config.API_METADATA.get("summary"),
            "license": config.API_METADATA.get("license"),
            "version": config.API_METADATA.get("version"),
            "datasets_local": utils.get_local_dirs(
                entries={'images', 'annotations'}
            ),
            "datasets_remote": utils.get_remote_dirs(
                entries={'images', 'annotations'}
            ),
            "models_local": utils.get_local_dirs(
                config.MODELS_PATH, entries={model_name}
            ),
            "models_remote": utils.get_remote_dirs(
                entries={model_name}
            ),
        }
        logger.debug("Package model metadata: %s", metadata)
        return metadata
    except Exception as err:
        logger.error("Error calling GET 'metadata': %s", err, exc_info=True)
        raise HTTPException(reason=err) from err


@utils.predict_arguments(schema=schemas.PredArgsSchema)
def predict(accept='application/json', **options):
    """Performs model prediction from given input data and parameters.

    Arguments:
        accept -- Response parser type, default is json.
        **options -- keyword arguments from PredArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        The predicted model values (dict or str) or files.
    """
    try:
        # handle input file (path from a browsing webargs field)
        tmp_filepath = Path(options['input_file'].filename)
        input_filepath = Path(config.DATA_PATH,
                              options['input_file'].original_filename)
        if input_filepath.suffix != ".npy":
            raise ValueError(
                f"Selected input '{input_filepath.name}' is not a .npy!"
            )
        shutil.copy(tmp_filepath, input_filepath)
        options = {**{"input_filepath": input_filepath}, **options}

        for k, v in options.items():
            logger.info(f"POST 'predict' argument - {k}:\t{v}")
        result = aimodel.predict(**options)

        # delete copied image
        input_filepath.unlink()

        logger.info("POST 'predict' result: %s", result)
        logger.debug("POST 'predict' returning content_type for: %s",
                     accept)
        return responses.content_types[accept](result, **options)
    except Exception as err:
        logger.error("Error while running POST 'predict': %s",
                     err, exc_info=True)
        raise  # Reraise the exception after log


@utils.train_arguments(schema=schemas.TrainArgsSchema)
def train(**options):
    """Performs model training from given input data and parameters.

    Arguments:
        **options -- keyword arguments from TrainArgsSchema.

    Raises:
        HTTPException: Unexpected errors aim to return 50X

    Returns:
        Parsed history/summary of the training process.
    """
    # MLFlow experiment tracking requires setting environment variables
    # and getting/injecting necessary credentials
    if options['mlflow_username']:
        MLFLOW_TRACKING_USERNAME = options['mlflow_username']
        logger.info(
            f"MLFlow model experiment tracking via account."
            f"\nUsername: {MLFLOW_TRACKING_USERNAME}"
        )
        MLFLOW_TRACKING_PASSWORD = getpass.getpass()   # inject password

        os.environ['MLFLOW_TRACKING_USERNAME'] = MLFLOW_TRACKING_USERNAME
        os.environ['MLFLOW_TRACKING_PASSWORD'] = MLFLOW_TRACKING_PASSWORD
        os.environ['LOGNAME'] = MLFLOW_TRACKING_USERNAME

    try:
        for k, v in options.items():
            logger.info(f"POST 'train' argument - {k}:\t{v}")
        result = aimodel.train(**options)
        logger.info(f"POST 'train' result: {result}")
        return result
    except Exception as err:
        logger.error("Error while running 'POST' train: %s",
                     err, exc_info=True)
        raise  # Reraise the exception after log
