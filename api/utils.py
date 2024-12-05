"""Utilities module for API endpoints and methods.
This module is used to define API utilities and helper functions.
"""
import logging
import os
# from pathlib import Path
import sys

from . import config

logger = logging.getLogger(__name__)
logger.setLevel(config.LOG_LEVEL)


def get_local_dirs(local_dir=config.DATA_PATH, entries: set = {}):
    """Call get_dirs for local directories"""
    return get_dirs(local_dir, entries)


def get_remote_dirs(remote_dir=config.REMOTE_PATH, entries: set = {}):
    """Call get_dirs for remote directories"""
    return get_dirs(remote_dir, entries)


def get_dirs(root_dir: str, entries: set = {}):
    """Utility to return a list of directories containing
    specific folder / file entries.
        - get_dirs(root_dir=config.DATA_PATH,
                   entries={'images', 'annotations'})
        - get_dirs(root_dir=config.REMOTE_PATH,
                   entries={'UNet.hdf5'})

    Arguments:
        root_dir (str): directory path to scan
        entries (set): entry patterns to search for, defaults to {}
    """
    dirscan = [
        root for root, dirs, files in os.walk(root_dir)
        if entries <= set(dirs) or entries <= set(files)
    ]
    return sorted(dirscan)


def get_default_path(directory_list: list, entries: set):
    """
    Utility to get the default path for marshmallow fields

    Args:
        directory_list (list): Path directories to be checked
        entries (set): dictionary of entry/ies to look for

    Returns:
        A default path for the marshmallow field in question
    """
    try:
        for d in directory_list:
            default_paths = get_dirs(d, entries=entries)

            if default_paths != []:
                return default_paths[0]

    except IndexError:
        return None


def generate_arguments(schema):
    """Function to generate arguments for DEEPaaS using schemas."""
    def arguments_function():  # fmt: skip
        print("Web args schema: ", schema)  # logger.debug
        return schema().fields
    return arguments_function


def predict_arguments(schema):
    """Decorator to inject schema as arguments to call predictions."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_predict_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema


def train_arguments(schema):
    """Decorator to inject schema as arguments to perform training."""
    def inject_function_schema(func):  # fmt: skip
        get_args = generate_arguments(schema)
        sys.modules[func.__module__].get_train_args = get_args
        return func  # Decorator that returns same function
    return inject_function_schema
