
import keras_core as keras


class RFFLayer(keras.layers.Layer):
    def __init__(self, num_features, ard: bool=False):
        super().__init__()
        self.num_features = num_features
        self.ard = ard

    def build(self, input_shape):
        self.omega = self.add_weight(
            shape=(input_shape[-1], self.num_features),
            initializer=keras.initializers.RandomNormal(stddev=1.0),
            trainable=False
        )
        self.length_scale = self.add_weight(
            shape=() if self.ard is False else (input_shape[-1],),
            initializer="random_normal",
            trainable=True,
            # constraint=keras.constraints.NonNeg()
        )
        self.variance = self.add_weight(
            shape=(),
            initializer="random_normal",
            trainable=True,
            # constraint=keras.constraints.NonNeg()
        )

    def call(self, inputs):
        kernel = self.omega / keras.ops.softplus(self.length_scale)
        inputs = keras.ops.matmul(inputs, kernel)
        inputs = keras.ops.hstack([
            keras.ops.sin(inputs),
            keras.ops.cos(inputs),
        ])
        inputs *= keras.ops.sqrt(self.variance**2 / self.num_features)
        return inputs

    def get_config(self):
        return {
            "num_features": self.num_features, 
            "ard": self.ard, 
        }