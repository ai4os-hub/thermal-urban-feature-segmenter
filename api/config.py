"""Module to define CONSTANTS used across the DEEPaaS Interface.

This module is used to define CONSTANTS used across the DEEPaaS API Interface.
The `config.py` file in the model_api package defines CONSTANTS related to
the model. By convention, CONSTANTS defined are in UPPER_CASE.
"""
import os
import logging
from importlib import metadata

# Necessary imports for api.__init__ code
from tufsegm_api.config import DATA_PATH  # noqa: F401
from tufsegm_api.config import REMOTE_PATH  # noqa: F401
from tufsegm_api.config import MODELS_PATH  # noqa: F401
from tufsegm_api.config import MODEL_TYPE  # noqa: F401
from tufsegm_api.config import MODEL_SUFFIX  # noqa: F401

# API name
API_NAME = os.getenv("API_NAME", default="tufsegm_api")

# Get AI metadata
API_METADATA = metadata.metadata(API_NAME)  # .json

# Fix metadata for emails from pyproject parsing
_EMAILS = API_METADATA["Author-email"].split(", ")
_EMAILS = map(lambda s: s[:-1].split(" <"), _EMAILS)
API_METADATA["Author-emails"] = dict(_EMAILS)

# Fix metadata for authors from pyproject parsing
_AUTHORS = API_METADATA.get("Author", "").split(", ")
_AUTHORS = [] if _AUTHORS == [""] else _AUTHORS
_AUTHORS += API_METADATA["Author-emails"].keys()
API_METADATA["Authors"] = sorted(_AUTHORS)

# logging level across API modules can be setup via API_LOG_LEVEL,
# options: DEBUG, INFO(default), WARNING, ERROR, CRITICAL
ENV_LOG_LEVEL = os.getenv("API_LOG_LEVEL", default="INFO")
LOG_LEVEL = getattr(logging, ENV_LOG_LEVEL.upper())
