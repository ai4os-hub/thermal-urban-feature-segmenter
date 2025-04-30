#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Fahimehs changes:
#   2- Changed the model_dir to a fixed for saving the model. Beacuse in 
#   each round the model should be saved in the same directory
#   3- The nvflare.client was add to the train_UNet script and mlflowwriter from nvflare
#   4- The other changes are numbered in the script



"""
Script for training U'Net model from segmentation-models toolbox

Author: Leon Klug, Elena Vollmer
Date: 26.06.2023
"""
# import built-in dependencies
import argparse
import copy
from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path, PosixPath
import sys
import time
import warnings
from tensorflow.keras.callbacks import EarlyStopping
import subprocess
# suppress messages from tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message="Passing (type, 1) or '1type' as a synonym of type is deprecated",
)


# import external dependencies
import git
import keras
import numpy as np
import tensorflow as tf
#tf.debugging.set_log_device_placement(True)
import tensorflow_addons as tfa

from keras.layers import Input, Conv2D
from evaluate import evaluate
#Some options such that TF does not allocate all the GPU resources
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.9
session = tf.compat.v1.Session(config=config)

#List the available GPUs and set XLA_FLAGS environment variable for TensorFlow
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs are available:")
    for gpu in gpus:
        print(" ", gpu)        
else:
    print("No GPUs are available.")

# (1) import nvflare client API
import nvflare.client as flare
from nvflare.client.tracking import MLflowWriter
from fedprox_loss import TFFedProxLoss



# import module dependencies
from tufseg.scripts.configuration import (
    init_temp_conf,
    update_conf,
    cp_conf,
    _default_config,
)

config = init_temp_conf()
from tufseg.scripts.segm_models._utils import (
    configure_logging,
    ImageProcessor,
    MaskProcessor,
)

# Very important! Enables Usage of segmentation models library
os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm

logging.getLogger("h5py").setLevel(logging.ERROR)
# --------------------------------------
_logger = logging.getLogger("train")
log_file = Path()

NUM_CLASSES = (
    len(config["data"]["masks"]["labels"]) + 1
)  # to account for the background class

# this fucntion logs metrics of form key, value:list of float
class GetKeyValuePairs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split("=")
            # use int if value is a digit, otherwise stay with string
            if value.isdigit():
                value = int(value)
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = value


class UpdateLocalWeightsCallback(tf.keras.callbacks.Callback):
    """
    This callback is called to make sure the local weights of the model are updated
    for FedProxLoss after each epoch.
    """

    def __init__(self):
        super().__init__()

    def on_epoch_end(self, epoch, logs=None):
        if (
            self.model.loss.local_model_weights
            == self.model.trainable_variables
        ):
            print(
                " the local model weights are updated after one epoch of training"
            )


def parse_args():
    parser = argparse.ArgumentParser(description="Training model")
    parser.add_argument(
        "-dst",
        "--dst-model-dir",
        dest="model_root",
        required=False,
        help="Path to destination directory to save model to.",
    )
    parser.add_argument(
        "-src",
        "--src-data-dir",
        dest="data_root",
        help="(Optional) Path to source directory of all data. "
        "If not provided, value from temp config (defined during setup) "
        "will be used.",
    )
    parser.add_argument(
        "-split",
        "--split-data-dir",
        dest="split_root",
        help="(Optional) Path to directory of train / test split. "
        "If not provided, value from temp config (defined during setup) "
        "will be used.",
    )
    parser.add_argument(
        "-ch",
        "--channels",
        dest="channels",
        type=int,
        default=3,
        choices=[4, 3],
        help="Whether to process the data as 3 channels "
        "(greyRGB+T+T) or keep the 4 channels (RGBT).",
    )
    parser.add_argument(
        "-proc",
        "--processing",
        dest="processing",
        type=str,
        default="basic",
        choices=["basic", "vignetting", "retinex_unsharp"],
        help="Whether to apply additional filters to the data (retinex unsharp, "
        "vignetting removal) or keep as basic (RGBT).",
    )
    parser.add_argument(
        "--only-tir",
        dest="only_tir",
        action="store_true",
        help="Whether to use only thermal image (not RGB). If the flag is added, "
        "the inputs will automatically be 3 channels and the selected "
        "processing option is applied.",
    )
    parser.add_argument(
        "-cfg",
        "--cfg-options",
        dest="cfg_options",
        nargs="+",
        action=GetKeyValuePairs,
        help="Specify training cfg options in the following manner: "
        '"seed=1000 lr=0.001 epochs=2 batch_size=2". Without these, the '
        "listed examples will be chosen as defaults",
    )
    parser.add_argument(
        "--default-log",
        dest="default_log",
        action="store_true",
        help="Add flag to log in default way instead of with configured logging.",
    )

    # Combine log level options into one with a default value
    log_levels_group = parser.add_mutually_exclusive_group()
    log_levels_group.add_argument(
        "--quiet",
        dest="log_level",
        action="store_const",
        const=logging.WARNING,
        help="Show only warnings.",
    )
    log_levels_group.add_argument(
        "-v",
        "--verbose",
        dest="log_level",
        action="store_const",
        const=logging.INFO,
        help="Show verbose log messages (info).",
    )
    log_levels_group.add_argument(
        "-vv",
        "--very-verbose",
        dest="log_level",
        action="store_const",
        const=logging.DEBUG,
        help="Show detailed log messages (debug).",
    )
    log_levels_group.set_defaults(log_level=logging.WARNING)
    return parser.parse_args()


def main(
    data_root: Path = None,
    split_root: Path = None,
    channels: int = 4,
    processing: str = "basic",
    only_tir: bool = False,
   
    default_log: bool = False,
    log_level=logging.WARNING,
    mu=0.00001
):
    
    # (2) initialize nvflare
    flare.init()

    # create model timestamp folder
    working_dir = os.getcwd()
    model_folder = "local_model"
    model_dir = os.path.join(working_dir, model_folder)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    mlflow_writer = MLflowWriter()

    # set up logging with log file in model directory
    global log_file
    log_file = Path(model_dir, "log.log")
    if default_log:
        logging.basicConfig(
            level=log_level,
            datefmt="%Y-%m-%d %H:%M",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(),
            ],
        )
    else:
        configure_logging(_logger, log_level, log_file)
    _logger.info("---TRAINING LOG---")

    # check for only_tir flag, which requires 3 channel inputs
    if channels != 3 and only_tir == True:
        _logger.info(
            f'Flag "only_tir" has been set, so only 3 channel inputs will be used. '
            f"Changing currently defined channel value of {channels} to 3..."
        )
        channels = 3

    # if only the data root is provided, make sure the split_root is defined as the
    # default config value, so that no previous definition in the temp config file
    if data_root is not None and split_root is None:
        split_root = _default_config["split_root"]
        print(split_root)

    # log local function arguments, including the user provided ones
    main_args = locals()
    for k, v in main_args.items():
        _logger.info(f"Parsed user argument - {k}:\t{v}")

    # update config file with those local arguments that aren't = None
    update_conf(
        conf=config,
        params={k: v for k, v in main_args.items() if v is not None},
    )

    
    client_id = flare.get_site_name()
    
    # load training and testing data
    _logger.info(
        f"Load data into {channels} channel images and process with "
        f"'{processing}' option."
    )
    X_train, y_train, X_test, y_test = load_data(client_id)

    # onehot encode y data (masks)
    y_train_onehot = np.asarray(tf.one_hot(y_train, NUM_CLASSES))
    y_test_onehot = np.asarray(tf.one_hot(y_test, NUM_CLASSES))

    # train
    _logger.info(f"Model cfg options:\n{config['model']}")
    # fahimeh:
    N = X_train.shape[-1]
    # Define the model
    _logger.info("Loading model backbone...")

    cfg = config["train"]
    _logger.info(f"Training cfg options:\n{cfg}")
    loss_name = cfg["loss"]["name"]
    try:
        loss_function = getattr(tfa.losses, loss_name)
    except AttributeError:
        raise ValueError(
            f"Loss function '{loss_name}' not found in tensorflow_addons.losses!"
        )

    optimizer_name = cfg["optimizer"]["name"]
    try:
        optimizer = getattr(tf.keras.optimizers, optimizer_name)
    except AttributeError:
        raise ValueError(
            f"Optimizer {optimizer_name} not found in tensorflow.keras.optimizers!"
        )

    _logger.info(
        f"Compiling model with '{loss_name}' loss function and '{optimizer_name}' optimizer."
    )
    tf.random.set_seed(cfg["seed"])
    learning_rate = cfg["optimizer"]["lr"]
    alpha = cfg["loss"]["alpha"]
    gamma = cfg["loss"]["gamma"]
    metrics = [eval(v) for v in config["eval"]["SM_METRICS"].values()]
    
    SIZE_H = config["data"]["loader"]["SIZE_H"]
    SIZE_W = config["data"]["loader"]["SIZE_W"]
    # Define number of channels, while influences training
    N = X_train.shape[-1]
    if N == 3:
        model = sm.Unet(
            config["model"]["backbone"],
            encoder_weights="imagenet",
            classes=NUM_CLASSES,
            input_shape=(SIZE_H, SIZE_W, N),
        )
    else:
        _logger.info(
            f"Channel count of {N} != 3. Adapting UNet by including a fitting first layer..."
        )
        base_model = sm.Unet(
            backbone_name=config["model"]["backbone"],
            encoder_weights="imagenet",
            classes=NUM_CLASSES,
        )

        inp = Input(shape=(SIZE_H, SIZE_W, N))
        layer_1 = Conv2D(3, (1, 1))(
            inp
        )  # map N channels data to 3 channels
        out = base_model(layer_1)



        model = keras.models.Model(
            inputs=inp, outputs=out, name=base_model.name
        )
    metrics = [eval(v) for v in config["eval"]["SM_METRICS"].values()]
    model.compile(
        optimizer=optimizer(learning_rate=learning_rate),
        loss=loss_function(alpha=alpha, gamma=gamma),
        metrics=metrics,
    )

    early_stopping= EarlyStopping( monitor="val_precision", 
    patience=3,
    verbose=1,
    restore_best_weights=True 
    )

    model_path = Path(model_dir, config["model"]["type"] + ".h5py")

    model_info = flare.receive()

    total_rounds = model_info.total_rounds

    #create a dict of the parameters that should be tracked
    model_config = copy.copy(config)
    
    model_params = {
        **model_config['model'], 
        'classes': model_config['data']['masks']['labels'],
        **model_config['data']['loader'],
        **model_config['train']
    }
    rounds = {"rounds": total_rounds}



    mlflow_writer.log_params(model_params)
    
    @flare.train
    def train(input_model=None):
        """
        Train UNet model with training images (X_train) and masks (y_train)
        and user-defined parameters.
        Use test dataset only to visualise performance before final evaluation.
        X data can be 3 channel or of different dimension (UNet will automatically be adapted).

        :param X_train: train images
        :param y_train_onehot: corresponding onehot encoded train masks
        :param X_test: test images
        :param y_test_onehot: corresponding onehot encoded test masks
        :param model_path: (Path) path to which model will be saved
        :return: model saved to provided path
        """
        nvconfig = flare.get_config() 
        print(nvconfig)        
        
        start = time.time()
        # Check that provided path can be used for saving model
        assert (
            type(model_path) == PosixPath
            and model_path.suffix == ".h5py"
        ), f"Provided model path '{model_path}' is not a Path ending in '.h5py'!"

        # (5) loads model from NVFlare
        for k, v in input_model.params.items():
            model.get_layer(k).set_weights(v)
        _logger.info("Training model...")

        local_model_weights = model.trainable_variables
        # (5) We deep copy the global model weights to avoid changing during training
        global_model_weights = copy.deepcopy(
            model.trainable_variables
        )

        # (6) Define the loss function using the fedproxloss class
        fedproxloss = TFFedProxLoss(
            local_model_weights, global_model_weights, mu, loss_function(alpha=alpha, gamma=gamma)
        )

        update_local_weights_callback = UpdateLocalWeightsCallback()


        combined_mean_scores_global,_=evaluate(model,config,X_test,y_test)

        metrics = [eval(v) for v in config["eval"]["SM_METRICS"].values()]


        model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=fedproxloss,
            metrics=metrics,
        )

        history = model.fit(
            X_train,
            y_train_onehot,
            batch_size=cfg["batch_size"],
            verbose=2,
            epochs=cfg["epochs"],
            validation_data=(X_test, y_test_onehot),
            callbacks=[CustomEpochLogger()],
        )
        

        combined_mean_scores_global = {f"{key}_global": value for key, value in combined_mean_scores_global.items()}


        mlflow_writer.log_metrics(
           metrics=combined_mean_scores_global , step=input_model.current_round
        )
        duration = time.time() - start
        _logger.info(
            f"Elapsed time during model training:\t{round(duration / 60, 2)} min"
        )

        # Save trained Model
        model.save_weights(model_path)
        #model.save(model_dir)
        _logger.info(f"Model saved to '{model_path}'.")
        save_model_info(model=model, model_dir=model_dir)

        # save json config to model directory
        # cp_conf(model_dir)
        _logger.info(
            f"Saved configuration of training run to {model_dir}"
        )

        # (3) send back the model to nvflare server
        output_model = flare.FLModel(
            params={
                layer.name: layer.get_weights()
                for layer in model.layers
            },
            metrics= combined_mean_scores_global,
        )
        return output_model

    # (4) gets FLModel from NVFlare
    while flare.is_running():
        input_model = flare.receive()
        current_round = input_model.current_round
        total_rounds = input_model.total_rounds

        print(f"Currently in round {current_round} out of a total of {total_rounds} rounds")

        # (optional) print system info
        system_info = flare.system_info()
        print(f"NVFlare system info: {system_info}")

        gpu_devices = tf.config.list_physical_devices('GPU')
        if gpu_devices:
            details = tf.config.experimental.get_device_details(gpu_devices[0])
            print(details)
            more_details = tf.sysconfig.get_build_info()
            print(more_details)

            message = subprocess.check_output('nvidia-smi')
            print(message) 

        #Show which client uses which GPU
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print("GPUs are available:")
            for gpu in gpus:
                print(" ", gpu)        
        else:
            print("No GPUs are available.")
    
        train(input_model=input_model)

        
        sys_info = flare.system_info()
        # LOG SOME INFO ABOUT THE CLIENT IN THE MLFLOW AS TAGS 
        mlflow_writer.set_tags(sys_info)



def load_data(site):
    """
    Load and process data according to filters in utils.py

    :return: X_train, y_train, X_test, y_test
    """
    if site=="site-1":
        data_path = "/hkfs/home/project/hk-project-test-p0023500/mp9809/datasets/dataset_MU/"
        #"/hkfs/home/project/hk-project-test-p0021801/uvecw/datasets/dataset_MU/"
    else:
        data_path = "/hkfs/home/project/hk-project-test-p0023500/mp9809/datasets/dataset_KA/"
        #"/Users/leo/Desktop/dataset_KA/"
    
    start = time.time()
    config['data_root'] =  data_path
    img_proc = ImageProcessor(config )
    print(config)
    X_train = img_proc.process_images(root="train")
    X_test = img_proc.process_images(root="test")

    mask_proc = MaskProcessor(config)
    y_train = mask_proc.load_masks(root="train")
    y_test = mask_proc.load_masks(root="test")

    duration = time.time() - start
    _logger.info(
        f"Elapsed time during data loading:\t{round(duration / 60, 2)} min"
    )
    return X_train, y_train, X_test, y_test


class CustomEpochLogger(tf.keras.callbacks.Callback):
    """
    Custom training logger to save epoch information to logging file
    """

    def __init__(self):
        super().__init__()
        self.current_batch = 0  # Initialize the batch number

    def on_epoch_begin(self, epoch, logs=None):
        self.current_batch = 0
        self.epoch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self.current_batch += 1

    def on_epoch_end(self, epoch, logs=None):
        epoch_duration = time.time() - self.epoch_start_time
        log_message = (
            f'\n--- Epoch {epoch + 1}/{self.params["epochs"]} ---\n'
        )

        log_message += (
            f'Batch: {self.current_batch}/{self.params["steps"]} --- '
            f"Time: {round(epoch_duration)}s/epoch - "
            f'{round((epoch_duration / self.params["steps"]) * 1000)}ms/step\n'
        )

        train_logs = {
            k: v for k, v in logs.items() if "val_" not in k
        }
        log_message += f'Training: {" - ".join([f"{k}: {round(v, 4)}" for k, v in train_logs.items()])}\n'

        val_logs = {k: v for k, v in logs.items() if "val_" in k}
        log_message += f'Validation: {" - ".join([f"{k}: {round(v, 4)}" for k, v in val_logs.items()])}\n'

        # Log to file
        with open(log_file, "a") as f:
            f.write(log_message)


def save_model_info(model, model_dir: Path):
    """
    Save relevant model information to model folder

    :param model: trained tensorflow model
    :param model_dir: (Path) directory to model outputs
    """
    # specify paths
    summary_path = Path(model_dir, "model_summary.txt")
    config_path = Path(model_dir, "model_config.json")

    # write to the files
    with open(summary_path, "w") as summary_file:
        stdout = sys.stdout
        sys.stdout = summary_file # Redirect print statements to the file
        model.summary()  # Print the summary to the file
        
        sys.stdout.close()
        sys.stdout = stdout

    _logger.info(f"Model summary saved to '{summary_path}'")

    with open(config_path, "w") as json_file:
        json.dump(
            model.get_config(), json_file, indent=4
        )  # Indent for better readability

    _logger.info(f"Model configuration saved to '{config_path}'")


if __name__ == "__main__":
    args = parse_args()
    main(
        data_root=args.data_root,
        split_root=args.split_root,
        channels=4,
        processing=args.processing,
        only_tir=args.only_tir,
        default_log=args.default_log,
        log_level=args.log_level,
    )
# nvflare simulator -n 2 -t 1 ./jobs/tensorflow_mlflow -w client_api_workspace
