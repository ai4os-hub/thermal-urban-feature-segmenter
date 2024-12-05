"""Module to define CONSTANTS used across the AI-model package.

This module is used to define CONSTANTS used across the AI-model package.
By convention, the CONSTANTS defined in this module are in UPPER_CASE.
"""
import logging
import os
import os.path as osp
from pathlib import Path

BASE_PATH = Path(__file__).resolve(strict=True).parents[1]

# Path definition for data folder
DATA_PATH = os.getenv("DATA_PATH", default=osp.join(BASE_PATH, "data"))
DATA_PATH = Path(DATA_PATH)
# Path definition for the pre-trained models
MODELS_PATH = os.getenv("MODELS_PATH", default=osp.join(BASE_PATH, "models"))
MODELS_PATH = Path(MODELS_PATH)

MODEL_TYPE = "UNet"
MODEL_SUFFIX = ".hdf5"

# Remote (rshare) paths for data and models
REMOTE_PATH = os.getenv("REMOTE_PATH", default="/storage/tufsegm")
REMOTE_DATA_PATH = os.getenv("REMOTE_DATA_PATH",
                             default=osp.join(REMOTE_PATH, "data"))
REMOTE_DATA_PATH = Path(REMOTE_DATA_PATH)
REMOTE_MODELS_PATH = os.getenv("REMOTE_MODELS_PATH",
                               default=osp.join(REMOTE_PATH, "models"))
REMOTE_MODELS_PATH = Path(REMOTE_MODELS_PATH)

# Define submodule name and path
SUBMODULE_NAME = 'TUFSeg'
SUBMODULE_PATH = Path(BASE_PATH, SUBMODULE_NAME, 'tufseg')

# configure logging:
# logging level across various modules can be setup via USER_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("USER_LOG_LEVEL", "INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())

if LOG_LEVEL == 10:
    VERBOSITY = "-vv"
elif LOG_LEVEL == 20:
    VERBOSITY = "-v"
else:
    VERBOSITY = "--quiet"

# Data limits on node to uphold
LIMIT_GB = int(os.getenv("LIMIT_GB", default="20"))
DATA_LIMIT_GB = int(os.getenv("DATA_LIMIT_GB", default="15"))

# Remote MLFlow server
MLFLOW_REMOTE_SERVER = "https://mlflow.cloud.ai4eosc.eu/"
MLFLOW_EXPERIMENT_NAME = SUBMODULE_NAME
