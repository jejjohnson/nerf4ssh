import jax.numpy as jnp
from jaxtyping import Array


def psnr(x_mse: Array) -> Array:
    return -10 * jnp.log(2 * x_mse)
