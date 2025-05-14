#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluate trained models
"""

import logging
import os


import warnings

# suppress messages from tf
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message="y_pred contains classes not in y_true",
)

# import external dependencies
import numpy as np
from sklearn import metrics as skmetrics
import tensorflow as tf
from tqdm import tqdm

from tufseg.scripts.segm_models._utils import (
 
    ImageProcessor,
    MaskProcessor,
)

# --------------------------------------
_logger = logging.getLogger("evaluate")

config: dict

 


def evaluate(model,config,X_test,y_test):
    """
    Coordinate model evaluation with user inputs
    and save evaluation results to a .json in the model directory
    """
 
    # evaluate
    _logger.info("Evaluate model on TEST dataset...")
   
    combined_mean_scores, combined_class_mean_scores = (
        evaluate_sklearn_imagewise(X_test, y_test, model=model,config=config)
    )

    return combined_mean_scores, combined_class_mean_scores




def evaluate_sklearn_imagewise(X_test, y_test, model,config):
    """
    Evaluate trained model using test dataset
    by inferring on images one at a time, calculating select scikit-learn metrics for each
    and combining the results to average scores for the whole test dataset.

    :param X_test: test images
    :param y_test: test masks
    :param model: loaded model
    :return: mean_scores: (dict) evaluation metrics applied to model
             mean_classwise_scores: (dict) class-wise evaluation metrics (excluding background)
    """
    _logger.info(
        "Evaluate model by evaluating test images individually and combining scores."
    )

    sklearn_metrics = config["eval"]["SKLEARN_METRICS"]
    total_scores = {metric: [] for metric in sklearn_metrics.keys()}

    classes = config["data"]["masks"]["labels"]
    classes.insert(0, "background")
    classwise_scores = {
        metric: {label: [] for label in classes}
        for metric in sklearn_metrics.keys()
    }

    for test_img, test_mask in tqdm(zip(X_test, y_test)):
        # Prediction
        prediction = model.predict(
            np.expand_dims(test_img, axis=0), verbose=0
        )
        class_predictions = np.argmax(prediction, axis=-1)
        class_predictions = np.squeeze(class_predictions)

        # Evaluate each of the metrics with scikit learn functions and their parameters
        for metric_name, metric_attrib in sklearn_metrics.items():
            function = getattr(skmetrics, metric_attrib["func"])
            metric_params = metric_attrib["params"]

            score = function(
                test_mask.flatten(),
                class_predictions.flatten(),
                **metric_params,
            )
            total_scores[metric_name].append(score)

            # skip weighted and balanced eval functions for class-wise evaluation
            if (
                "weighted" in metric_name.lower()
                or "balanced" in metric_name.lower()
            ):
                continue

            # evaluate class-wise
            for class_id, class_name in enumerate(classes):
                class_mask_true = test_mask == class_id
                class_mask_pred = class_predictions == class_id

                # evaluate only those images that actually contain or predict the class
                if np.any(class_mask_true) or np.any(class_mask_pred):
                    metric_singleclass_params = metric_params.copy()
                    if "average" in metric_params.keys():
                        metric_singleclass_params["average"] = (
                            "binary"
                        )

                    class_score = function(
                        class_mask_true.flatten(),
                        class_mask_pred.flatten(),
                        **metric_singleclass_params,
                    )
                    classwise_scores[metric_name][class_name].append(
                        class_score
                    )

    mean_scores = {
        f"mean_{k}": np.mean(v) for k, v in total_scores.items()
    }

    print(
        "Combined image-wise evaluation results (scikit-learn metrics):"
    )
    for key, val in mean_scores.items():
        print(
            f"{key.ljust(max(len(k) for k in mean_scores.keys()))}: {round(val, 4)}"
        )

    mean_classwise_scores = {}
    for metric_name in sklearn_metrics.keys():
        if (
            "weighted" not in metric_name.lower()
            and "balanced" not in metric_name.lower()
        ):
            classwise_metric_name = f"mean_{metric_name}_classwise"
            classwise_scores_dict = {
                class_name: np.mean(scores)
                for class_name, scores in classwise_scores[
                    metric_name
                ].items()
            }
            mean_classwise_scores[classwise_metric_name] = (
                classwise_scores_dict
            )

    print("------------------- class-wise evaluation results:")
    for key1, val1 in mean_classwise_scores.items():
        print(f"{key1}")
        for key2, val2 in val1.items():
            print(
                f"{key2.ljust(max(len(k) for k in val1.keys()))}: "
                f"{round(val2, 4)}"
            )

    return mean_scores, mean_classwise_scores
