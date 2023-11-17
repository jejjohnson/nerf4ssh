import keras_core as keras


@keras.saving.register_keras_serializable()
def psnr(y_true, y_pred):
    mse = keras.losses.mean_squared_error(y_true, y_pred)
    return - 10 * keras.ops.mean(keras.ops.log10(mse), axis=-1)
