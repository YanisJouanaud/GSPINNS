from typing import TypedDict, Union, NamedTuple
from jax import Array
from jaxtyping import Float, Int, PyTree
from jax.typing import ArrayLike
import jax.numpy as jnp

Data_Type = TypedDict("Data_Type", {0:Array, 1:Array})

class Data_Container(NamedTuple):
    time: Array
    data: Data_Type
    Tmax: Float

class Pinn_Container(NamedTuple):
    lambda0: Union[Float, tuple[Float, Float, Float]]
    lambda1: Float
    lambda2: Float
    params: Array

class Metrics_Container(NamedTuple):
    errprec: Float
    err: Float

class Proximal_Container(NamedTuple):
    prox_coef:Float

class Condition_Container(NamedTuple):
    errMax:Float
    iterMax:Int
    epochs: Int
