import keras_core as keras



class SineInitializers(keras.initializers.Initializer):
    def __init__(self, c, omega, layer_type: str="hidden"):
        self.c = c
        self.omega = omega
        self.layer_type = layer_type
    def __call__(self, shape, dtype=None):
        if self.layer_type == "first":
            limit = 1 / shape[0]
        elif self.layer_type == "hidden":
            limit = keras.ops.sqrt(self.c / shape[0]) / self.omega
        elif self.layer_type == "last":
            limit = keras.ops.sqrt(self.c / shape[0])
            
        return keras.random.uniform(shape, -limit, limit)

    def get_config(self):
        return {"c": self.c, "omega": self.omega, "layer_type": self.layer_type}

class SirenLayer(keras.layers.Layer):
    def __init__(self, units=32, omega=1.0, c=6.0, layer_type: str="hidden"):
        super().__init__()
        self.units = units
        self.omega = omega
        self.c = c
        self.layer_type = layer_type
        self.activation = True if layer_type in ["first", "hidden"] else False

    def build(self, input_shape):
        self.w = self.add_weight(
            shape=(input_shape[-1], self.units),
            initializer=SineInitializers(c=self.c, omega=self.omega, layer_type=self.layer_type),
            trainable=True,
        )
        limits = keras.ops.sqrt(1.0 / input_shape[-1])
        self.b = self.add_weight(
            shape=(self.units,), 
            initializer=keras.initializers.RandomUniform(
                minval=-limits, 
                maxval=limits
            ), 
            trainable=True
        )

    def call(self, inputs):
        outputs =  keras.ops.matmul(inputs, self.w) + self.b
        if self.activation:
            outputs = keras.ops.sin(self.omega * outputs)
        return  outputs
    def get_config(self):
        return {"c": self.c, "omega": self.omega, "layer_type": self.layer_type, "units": self.units}