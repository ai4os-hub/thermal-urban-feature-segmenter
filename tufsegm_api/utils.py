"""Package to create datasets, pipelines and other utilities.

This module is used to define all the functions needed to
operate the methods defined at `__init__.py`.
"""
import json
import logging
from math import floor
import mlflow
import mlflow.tensorflow
import os
from pandas.io.json._normalize import nested_to_record
from pathlib import Path
import shutil
import subprocess
import tensorflow as tf
import time
import threading
import zipfile

import tufsegm_api.config as cfg
from tufseg.scripts.segm_models._utils import ModelLoader
from tufseg.scripts.configuration import read_conf

logger = logging.getLogger(__name__)
logger.setLevel(cfg.LOG_LEVEL)


PROJ_LIM_OPTIONS = {
    "BASE": {"LIMIT": cfg.LIMIT_GB, "PATH": cfg.BASE_PATH},
    "DATA": {"LIMIT": cfg.DATA_LIMIT_GB, "PATH": cfg.DATA_PATH}
}


class DiskSpaceExceeded(Exception):
    """Raised when disk space is exceeded."""
    pass


class SubprocessError(Exception):
    """Raised when disk space is exceeded."""
    pass


def configure_api_logging(logger, log_level: int):
    """Define basic logging configuration

    :param logger: logger
    :param log_level: User defined input
    """
    log_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Define logging to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)
    # logger.setLevel(log_level)


def copy_remote(frompath, topath):
    """Copies remote (e.g. NextCloud) folder/file in your local deployment or
    vice versa for example:
        - `copy_remote('/storage/data/images', '/srv/myapp/data/images')`
    Ensures deployment node space isn't being exceeded during copying.

    Args:
        frompath (Path): The path to the file to be copied
        topath (Path): The path to the destination folder directory

    Raises:
        OSError: If the source isn't a directory
        FileNotFoundError: If the source file doesn't exist
    """
    frompath: Path = Path(frompath)
    topath: Path = Path(topath)
    topath_contents = set(topath.iterdir())

    log_disk_usage(f"Begin copying from '{frompath}' to '{topath}'...")
    # get absolute limit by comparing with available node space
    limit_gb = check_available_space()
    limit_bytes = floor(limit_gb * (1024 ** 3))    # convert to bytes

    try:
        if frompath.is_dir():

            for f in sorted(frompath.rglob("[!.]*")):

                if f.is_dir():
                    topath_folder = Path(topath, f.relative_to(frompath))
                    topath_folder.mkdir(parents=True, exist_ok=True)

                elif f.is_file():
                    file_size = f.stat().st_size
                    if get_disk_usage() + file_size >= limit_bytes:
                        raise DiskSpaceExceeded(
                            f"Copying file will exceed the disk space limit "
                            f"of {limit_gb} GB for '{cfg.BASE_PATH}' folder.")
                    shutil.copy(
                        f,
                        Path(topath, f.relative_to(frompath).parent)
                    )
                    log_disk_usage(f"Copied '{f}'")

                else:
                    raise FileNotFoundError

        elif frompath.is_file():

            file_size = f.stat().st_size
            if get_disk_usage() + file_size >= limit_bytes:
                raise DiskSpaceExceeded(
                    f"Copying file will exceed the disk space limit "
                    f"of {limit_gb} GB for '{cfg.BASE_PATH}' folder.")
            shutil.copy(frompath, topath)

        else:
            raise OSError

    except (OSError, FileNotFoundError) as e:
        logger.error(f"Error in copying from '{frompath}' to '{topath}'. "
                     f"Error: %s" % e)
        raise

    except DiskSpaceExceeded as e:
        logger.error(f"Disk space limit almost exceeded: {str(e)}.")

        delete_new_contents(topath_contents, set(topath.iterdir()))

        raise DiskSpaceExceeded(
            "You will need to free up some space on the node to download"
            " all the data or work in remote (/storage/) directories!")

    log_disk_usage("Copying complete")


def delete_new_contents(original_contents: set, current_contents: set):
    """Deletes newly copied files and folders

    Args:
        original_contents (set): existing files/folder paths before copying
        current_contents (set): files/folder paths after copying
    """
    new_contents = current_contents - original_contents
    logger.info(f"Deleting newly downloaded files/folders: "
                f"{len(new_contents)}")

    for item in new_contents:
        if item.is_dir():
            shutil.rmtree(item)
        else:
            item.unlink()


def unzip(zip_paths: list):
    """
    Unzipping files while staying below the deployment space limit.

    Args:
        zip_paths (list): .zip files to extract

    Raises:
        DiskSpaceExceeded: If available space exceeded during unzipping.
    """
    log_disk_usage(f"Begin unzipping {len(zip_paths)} .zip files. "
                   f"This may take a while...")

    limit_gb = check_available_space(PROJ_LIM_OPTIONS["DATA"])   # abs limit
    limit_bytes = floor(limit_gb * (1024 ** 3))   # convert to bytes

    for zip_path in zip_paths:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for file_info in zip_ref.infolist():

                f_size = file_info.file_size
                if get_disk_usage(cfg.DATA_PATH) + f_size >= limit_bytes:
                    raise DiskSpaceExceeded(
                        f"Unzipping will exceed the max allowed disk space "
                        f"of {limit_gb} GB for '{cfg.DATA_PATH}' folder."
                    )
            # unzip the file to its current directory
            logger.info(f"Unzipping '{zip_path}'")
            zip_ref.extractall(zip_path.parent)

        logger.info("Cleaning up zip file...")
        zip_path.unlink()
        log_disk_usage(f"Unzipped '{zip_path}'")

    log_disk_usage("Unzipping complete")


def setup(data_path: Path, test_size: int, save_for_view: bool = False):
    """
    General setup function for annotation processing and dataset split.
    Generates segmentation masks from annotation data and splits the data into
    test and training sets.

    Args:
        data_path: pathlib.Path to the data directory.
        test_size: int. Size of the test set in splitting of train test.

    Raises:
        FileNotFoundError: If the necessary files don't exist after setup.
        DiskSpaceExceeded: If available disk space was exceeded during setup.
    """
    log_disk_usage("Beginning setup")

    limit_gb = check_available_space(PROJ_LIM_OPTIONS["DATA"])

    try:
        # monitor disk space usage in the background
        monitor_thread = threading.Thread(target=monitor_disk_space,
                                          args=(limit_gb,), daemon=True)
        monitor_thread.start()

        setup_path = Path(cfg.SUBMODULE_PATH, 'scripts', 'setup', 'setup.sh')
        if not setup_path.is_file():
            raise FileNotFoundError(f"File '{setup_path}' does not exist!")

        setup_cmd = [
            "/bin/bash",
            str(setup_path),
            "-j", str(Path(data_path, 'annotations')),
            "-i", str(Path(data_path, 'images')),
            "--test-size", str(test_size),
            cfg.VERBOSITY
        ]
        if save_for_view:
            setup_cmd.insert(-1, "--save-for-view")

        run_bash_subprocess(setup_cmd)

    except DiskSpaceExceeded as e:
        logger.error(f"Disk space limit exceeded: {str(e)}")
        raise DiskSpaceExceeded(
            f"Setting up exceeds the maximum allowed disk space "
            f"of {limit_gb} GB for '{cfg.DATA_PATH}' folder."
        )

    if not set(os.listdir(data_path)) >= {"masks", "train.txt", "test.txt"}:
        raise FileNotFoundError(
            f"Data path '{data_path}' does not contain required "
            f"entries after setup!"
        )

    log_disk_usage("Setup complete")


def run_bash_subprocess(cmd: list, timeout: int = 1000):
    """
    Run bash script call via subprocess command
    while printing all outputs to the terminal

    Args:
        cmd -- list of command line arguments for subprocess call
        timeout -- int. Timeout in seconds for the subprocess command
    """
    logger.debug(f"Running subprocess command with arguments: '{cmd}'")

    # check available physical devices (GPU or CPU)
    if not tf.config.experimental.list_physical_devices('GPU'):
        timeout = timeout * 3
        logger.warning(f"No GPU devices detected, running on CPU. "
                       f"Extending timeout to {timeout} seconds.")

    try:
        process = subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)
        return_code = process.wait(timeout=timeout)

        # check return code to stop if bash script was forcefully exited
        if return_code == 0:
            logger.info("Bash script executed successfully.")
        else:
            process.terminate()
            raise SubprocessError(
                f"Error during execution of bash script '{cmd[1]}'. "
                f"Terminated with return code {return_code}.")

    except subprocess.TimeoutExpired:
        logger.error(f"Timeout during execution of bash script '{cmd[1]}'.")
        process.terminate()
        raise SubprocessError(
            f"Timeout during execution of bash script '{cmd[1]}'."
        )


def mlflow_logging(model_root: Path):
    """
    Logging model experiment to MLFlow server.

    Args:
        model_root (Path) -- Path to model folder
    """
    # set the MLflow server and backend and artifact stores
    mlflow.set_tracking_uri(cfg.MLFLOW_REMOTE_SERVER)

    # set the experiment name for all different runs
    mlflow.set_experiment(cfg.MLFLOW_EXPERIMENT_NAME)

    model_config_path = Path(model_root, "run_config.json")
    model_config = read_conf(model_config_path)
    model_params = {
        **model_config['model'],
        'classes': model_config['data']['masks']['labels'],
        **model_config['data']['loader'],
        **model_config['train']
    }

    model_loader = ModelLoader(model_config, model_root)
    model = model_loader.model

    with open(Path(model_root, "eval.json"), "r") as f:
        k1 = 'sklearn metrics - combined imagewise results'
        k2 = 'sklearn metrics - combined imagewise classwise results'

        model_metrics = json.load(f)
        model_metrics = {**model_metrics[k1], **model_metrics[k2]}
        model_metrics_flat = nested_to_record(model_metrics, sep=' ')
        model_metrics_flat = {
            k.replace('(', '- ').replace(')', ''): v
            for k, v in model_metrics_flat.items()
        }

    with mlflow.start_run(run_name=Path(model_root).name):
        mlflow.tensorflow.log_model(model, artifact_path='artifacts')
        mlflow.log_params(model_params)
        logger.info("MLFlow - logged training parameters.")
        mlflow.log_metrics(model_metrics_flat)
        logger.info("MLFlow - logged evaluation metrics.")

    return


# ###################################
# HELPER FUNCTIONS FOR UTIL FUNCTIONS
# ###################################


def monitor_disk_space(limit_gb):
    """
    Thread function to monitor disk space and check the current usage
    doesn't exceed the available disk space limit.

    Arguments:
        limit_gb (int): identified available disk space (in GB)

    Raises:
        DiskSpaceExceeded: If available space was exceeded while threading.
    """
    limit_bytes = limit_gb * (1024 ** 3)  # convert to bytes
    try:
        while True:
            time.sleep(5)

            stored_bytes = get_disk_usage()

            if stored_bytes >= limit_bytes:
                raise DiskSpaceExceeded(
                    f"Exceeded maximum allowed disk space of {limit_gb} GB "
                    f"for '{cfg.BASE_PATH}' (or a subfolder)."
                )
            else:
                leftover_gb = round(
                    (limit_bytes - stored_bytes) / (1024 ** 3), 2
                )
                logger.info(f"Leftover disk space: {leftover_gb} GB")

    except DiskSpaceExceeded as e:
        logger.error(f"Child thread terminating due to: {str(e)}")
        raise DiskSpaceExceeded


def check_available_space(
    proj_lim_option: dict = PROJ_LIM_OPTIONS["BASE"]
):
    """
    Check overall data limit on node and redefine limit if necessary.

    Args:
        proj_lim_option (dict): {"LIMIT": int, "PATH": pathlib.Path}

    Returns:
        available GB on node
    """
    project_limit_gb = proj_lim_option["LIMIT"]
    # get used project space and theoretically remaining available space
    project_used_gb = round(
        get_disk_usage(proj_lim_option["PATH"]) / (1024 ** 3), 2
    )
    project_available_gb = round(project_limit_gb - project_used_gb, 2)

    try:
        # get available space on entire node
        cmd = "df -h | grep 'overlay' | awk '{print $4}'"
        node_available_gb = float(subprocess.getoutput(cmd).split("G")[0])

    except ValueError:
        logger.info(
            f"ValueError: Node disk space not readable. "
            f"Using provided limit of {project_limit_gb} GB.")
        node_available_gb = project_limit_gb

    # redefine available project space (with a safety margin)
    safety = 2
    if node_available_gb <= safety:
        raise DiskSpaceExceeded(
            f"Available node disk space ({node_available_gb} GB) "
            f"below safety margin."
        )

    if node_available_gb - safety <= project_available_gb:
        logger.warning(
            f"Available node disk space ({node_available_gb} GB) is less "
            f"than the theoretically remaining reserved project space "
            f"({project_available_gb} GB). Limit will be reduced from "
            f"{project_limit_gb} GB to {project_used_gb + node_available_gb}"
            f"GB minus a safety margin of {safety} GB.")

        project_available_gb = node_available_gb - safety

    limit_gb = project_used_gb + project_available_gb
    logger.info(f"Absolute project limit is {round(limit_gb, 2)} GB.")
    return limit_gb


def get_disk_usage(folder: Path = cfg.BASE_PATH):
    """Get the current amount of bytes stored in the provided folder.
    """
    return sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())


def log_disk_usage(process_message: str):
    """Log used disk space to the terminal with a process_message description.
    """
    logger.info(
        f"{process_message}: Repository currently takes up "
        f"{round(get_disk_usage() / (1024 ** 3), 2)} GB."
    )
