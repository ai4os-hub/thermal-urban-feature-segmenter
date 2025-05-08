import tensorflow as tf


class TFFedProxLoss(tf.keras.losses.Loss):
    def __init__(
        self,
        local_model_weights,
        global_model_weights,
        mu,
        loss_fun,
        *args,
        **kwargs
    ):
        """
        Initialize the TFFedProxLoss class.

        Parameters:
        local_model_weights: the local model trainable weights
        global_model_weights: the global model trainable weights
        mu: The regularization parameter for the FedProx term.
        loss_fun:  The base loss function to be used
        (e.g., tf.keras.losses.SparseCategoricalCrossentropy).

        Returns
        A loss function that includes the FedProx regularization term.
        """
        super().__init__(*args, **kwargs)
        self.local_model_weights = local_model_weights
        self.global_model_weights = global_model_weights
        self.mu = mu
        self.loss_fun = loss_fun

    def call(self, y_true, y_pred):

        original_loss = self.loss_fun(y_true, y_pred)

        fedprox_term = (
            self.mu / 2
        ) * self.difference_model_norm_2_square(
            self.global_model_weights, self.local_model_weights
        )

        return original_loss + fedprox_term

    def difference_model_norm_2_square(
        self, global_model, local_model
    ) -> tf.Tensor:
        """
        Calculates the squared l2 norm of a model difference.
        Args:
            global_model: the trainable variables model broadcast by the server
            local_model: the trainable variables  of the local model

        Returns: the squared norm
        """

        model_difference = tf.nest.map_structure(
            lambda a, b: a - b, local_model, global_model
        )
        squared_norm = tf.square(
            tf.linalg.global_norm(model_difference)
        )
        return squared_norm