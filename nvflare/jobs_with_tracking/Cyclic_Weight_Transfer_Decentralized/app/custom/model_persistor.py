# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
from keras.layers import Input, Conv2D, Lambda
import keras

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
)
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.tf.utils import (
    flat_layer_weights_dict,
    unflat_layer_weights_dict,
)

# Fafa
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import tensorflow as tf
import segmentation_models as sm
import tensorflow_addons as tfa
from keras.layers import Input, Conv2D
import keras

from nvflare.apis.event_type import EventType
from nvflare.apis.fl_context import FLContext
from nvflare.app_common.abstract.model import (
    ModelLearnable,
    ModelLearnableKey,
    make_model_learnable,
)
from nvflare.app_common.abstract.model_persistor import ModelPersistor
from nvflare.app_opt.tf.utils import (
    flat_layer_weights_dict,
    unflat_layer_weights_dict,
)

# Fafa
import mlflow
import mlflow.keras
from tufseg.scripts.configuration import init_temp_conf


config = init_temp_conf()
SIZE_H = config["data"]["loader"]["SIZE_H"]
SIZE_W = config["data"]["loader"]["SIZE_W"]
cfg = config["train"]
alpha = cfg["loss"]["alpha"]
gamma = cfg["loss"]["gamma"]
optimizer_name = cfg["optimizer"]["name"]
optimizer = getattr(tf.keras.optimizers, optimizer_name)
learning_rate = cfg["optimizer"]["lr"]
alpha = cfg["loss"]["alpha"]
gamma = cfg["loss"]["gamma"]
loss_name = cfg["loss"]["name"]
loss_function = getattr(tfa.losses, loss_name)
NUM_CLASSES = len(config['data']['masks']['labels']) + 1


metrics = [eval(v) for v in config["eval"]["SM_METRICS"].values()]


class TFModelPersistor(ModelPersistor):
    def __init__(
        self, model: tf.keras.Model, N = 4, save_name="tf_model.ckpt"
    ):
        super().__init__()
        self.N = N  # Number of input channels=3 or 4
        self.save_name = save_name
        self.model = model
        if self.N > 3:
            base_model = self.model
            inp = Input(shape=(SIZE_H, SIZE_W, self.N))
            #inp = Input(shape=(SIZE_H, SIZE_W, N))
            layer_1 = Conv2D(3, (1, 1))(
                inp
            )  # map N channels data to 3 channels
            out = base_model(layer_1)

            #out = Lambda(lambda x: base_model(x), name='base_model_output')(layer_1) 



            self.model = keras.models.Model(
                inputs=inp, outputs=out, name=base_model.name
            )
        self.model.compile(
            optimizer=optimizer(learning_rate=learning_rate),
            loss=loss_function(alpha=alpha, gamma=gamma),
            metrics=metrics,
        )    

    def _initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_root = workspace.get_app_dir(fl_ctx.get_job_id())
        self._model_save_path = os.path.join(app_root, self.save_name)
        #self.mlflow_setup(art_full_path, experiment_name, experiment_tags, sites)

    def load_model(self, fl_ctx: FLContext) -> ModelLearnable:
        """Initializes and loads the Model.

        Args:
            fl_ctx: FLContext

        Returns:
            ModelLearnable object
        """

        if os.path.exists(self._model_save_path):
            self.logger.info("Loading server model and weights")
            self.model.load_weights(self._model_save_path)

        # get flat model parameters
        layer_weights_dict = {
            layer.name: layer.get_weights()
            for layer in self.model.layers
        }
        result = flat_layer_weights_dict(layer_weights_dict)

        model_learnable = make_model_learnable(result, dict())
        return model_learnable

    def handle_event(self, event: str, fl_ctx: FLContext):
        if event == EventType.START_RUN:
            self._initialize(fl_ctx)

    def save_model(
        self, model_learnable: ModelLearnable, fl_ctx: FLContext
    ):
        result = unflat_layer_weights_dict(
            model_learnable[ModelLearnableKey.WEIGHTS]
        )
        run_ids = fl_ctx.get_prop("RUN_IDS", default=None)
        print("Running ids are: %s" % run_ids)
        self.save_model_file(self._model_save_path, result, run_ids)

    def save_model_file(self, save_path: str, result, run_ids):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        # result = unflat_layer_weights_dict(model_learnable[ModelLearnableKey.WEIGHTS])
        for k in result:
            layer = self.model.get_layer(name=k)
            layer.set_weights(result[k])
        self.model.save_weights(save_path)

        input_arr = tf.random.uniform((1, SIZE_H, SIZE_W, self.N))

        import logging

        logging.getLogger("mlflow").setLevel(logging.DEBUG)
        logging.basicConfig(level=logging.DEBUG)
        custom_objects = {
            "Addons>SigmoidFocalCrossEntropy": tfa.losses.SigmoidFocalCrossEntropy(
                alpha=alpha, gamma=gamma
            ),
        }
        for metric_name, metric_instance in config["eval"][
            "SM_METRICS"
        ].items():
            custom_objects[metric_name] = eval(metric_instance)
        custom_objects["f1-score"] = sm.metrics.FScore()
        
        # save the model for each run_id such that it is available for each tracked site within mlflow
        #for key in run_ids:
        key= list(run_ids.keys())[0]
        print(f'the key of the mlflow rounds is {key}')
     
        with mlflow.start_run(run_id=run_ids[key]):
                    _ = self.model(input_arr)
                    mlflow.keras.log_model(
                        self.model,
                        custom_objects=custom_objects,
                        artifact_path="tf-models",
                    )