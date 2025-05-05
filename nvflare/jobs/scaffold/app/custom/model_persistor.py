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
import tensorflow_addons as tfa
from keras.layers import Input, Conv2D
import keras

os.environ["SM_FRAMEWORK"] = "tf.keras"
import segmentation_models as sm
sm.set_framework('tf.keras')

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
from tensorflow_addons.layers import GroupNormalization
config = init_temp_conf()
SIZE_H = config["data"]["loader"]["SIZE_H"]
SIZE_W = config["data"]["loader"]["SIZE_W"]

tf.random.set_seed(config["train"]["seed"])

def get_unet_with_groupnorm(model):
     
    # Replace all BatchNormalization layers with GroupNormalization
    input_tensor = model.input
    x = input_tensor
    
    def replace_layer(layer):
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            channels = layer.input_shape[-1]
            # Choose appropriate number of groups
            num_groups = min(32, channels)
            # Ensure number of channels is divisible by groups
            while channels % num_groups != 0:
                num_groups -= 1 
            return GroupNormalization(
                groups=num_groups,
                axis=-1,
                epsilon=0.001,
                center=True,
                scale=True
            )
        return layer
    
    # Create new model with replaced layers
    new_model = tf.keras.models.clone_model(
        model,
        clone_function=replace_layer
    )
    
    # Copy weights for non-BatchNorm layers
    for layer, new_layer in zip(model.layers, new_model.layers):
        if not isinstance(layer, tf.keras.layers.BatchNormalization):
            new_layer.set_weights(layer.get_weights())
    
    return new_model
class TFModelPersistor(ModelPersistor):
    def __init__(
        self, model: tf.keras.Model, N: int, save_name="tf_model.weights.h5", filter_id: str = None
    ):
        super().__init__(
            filter_id=filter_id,
        )
        self.N = N  # Number of input channels=3 or 4
        self.save_name = save_name
        self.model = model
        self._input_shape = (SIZE_H, SIZE_W, self.N)
        if self.N > 3:
            base_model = self.model
            base_model = get_unet_with_groupnorm(base_model)
            
            inp = Input(shape=(SIZE_H, SIZE_W, self.N))
            layer_1 = Conv2D(3, (1, 1))(
                inp
            )
            
            
            # map N channels data to 3 channels
            out = base_model(layer_1)
            self.model = keras.models.Model(
                inputs=inp, outputs=out, name=base_model.name
            )

            self.model.summary()
            

   

    def _initialize(self, fl_ctx: FLContext):
        workspace = fl_ctx.get_engine().get_workspace()
        app_root = workspace.get_app_dir(fl_ctx.get_job_id())
        self._model_save_path = os.path.join(app_root, self.save_name)
        self._best_model_save_path= os.path.join(app_root, 'best_global_model')
        self.best_model = None
        self.run_ids = fl_ctx.get_prop("RUN_IDS", default=None)

        

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
        if not self.model.built:
                self.model.build(input_shape=self.input_shape)
    
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
            
    def save_model(self, model_learnable: ModelLearnable, fl_ctx: FLContext):
        """Saves model.

        Args:
            model_learnable: ModelLearnable object
            fl_ctx: FLContext
        """
        result = unflat_layer_weights_dict(model_learnable[ModelLearnableKey.WEIGHTS])
        for k in result:
            layer = self.model.get_layer(name=k)
            layer.set_weights(result[k])
        self.model.save_weights(self._model_save_path)
      
