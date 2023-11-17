import numpy as np
import keras_core as keras


class RFFLayer(keras.layers.Layer):
    def __init__(
        self, 
        num_features, 
        ard: bool=False,
        length_scale_train: bool=False, 
        variance_train: bool=False
    ):
        super().__init__()
        self.num_features = num_features
        self.ard = ard
        self.length_scale_train = length_scale_train
        self.variance_train = variance_train
    def build(self, input_shape):
        self.omega = self.add_weight(
            shape=(input_shape[-1], self.num_features),
            initializer=keras.initializers.RandomNormal(stddev=1.0),
            trainable=False
        )
        self.length_scale = self.add_weight(
            shape=() if self.ard is False else (input_shape[-1],),
            initializer=keras.initializers.constant(get_default_scale(input_shape[-1])),
            trainable=self.length_scale_train,
            constraint=keras.constraints.NonNeg()
        )
        self.variance = self.add_weight(
            shape=(),
            initializer=keras.initializers.constant(1.0),
            trainable=self.variance_train,
            constraint=keras.constraints.NonNeg()
        )

    def call(self, inputs):
        kernel = (1.0 / self.length_scale) * self.omega
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
            "length_scale_train": self.length_scale_train,
            "variance_train": self.variance_train,
            
        }


def get_default_scale(input_dim, ard: bool=False):

    length_scale = np.sqrt(input_dim / 2.0)
    
    if ard:
        return np.asarray([length_scale] * input_dim)
        
    return length_scale