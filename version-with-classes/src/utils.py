from collections.abc import Callable
import jax
import jaxopt as jo
from jax import random, vmap
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
import matplotlib.pyplot as plt
from jaxtyping import Array, Int, PRNGKeyArray
from functools import partial
from typing import Union, Optional, NamedTuple, Any
from jinns.utils._pinn import PINN

class EmptyState(NamedTuple):
    """An empty state, we don't need to store anything for the proximal operator of l1 regularization."""
    pass

class PSnake(eqx.Module, strict=True):

    frequency: Array

    def __init__(
        self,
        init_alpha: Optional[Union[float, Array]]=10.,
        key: PRNGKeyArray=None,
    ):
        r"""**Arguments:**

        - `init_alpha`: The initial value $\alpha$ of the frequency.
            This should either be a `float` (default value is $10.$), or
            a JAX array of $\alpha_i$ values. The shape of such a JAX array
            should be broadcastable to the input.
        """

        self.frequency = jnp.asarray(init_alpha)

    @jax.named_scope("PSnake")
    def __call__(self, x: Array, key: PRNGKeyArray=None,) -> Array:
        r"""**Arguments:**

        - `x`: The input.

        **Returns:**

        A JAX array of the same shape as the input.
        """
        with jax.numpy_dtype_promotion("standard"), jax.numpy_rank_promotion("allow"):
            return x + (jnp.sin(self.frequency*x)**2)/self.frequency


def snake(x):
    a=5
    return x+(jnp.sin(a*x)**2)/a


class PSinc(eqx.Module, strict=True):

    damp: Array

    def __init__(
        self,
        init_alpha: Optional[Union[float, Array]]=1.,
        key: PRNGKeyArray=None,
    ):
        r"""**Arguments:**

        - `init_alpha`: The initial value $\alpha$ of the damping.
            This should either be a `float` (default value is $10.$), or
            a JAX array of $\alpha_i$ values. The shape of such a JAX array
            should be broadcastable to the input.
        """

        self.damp = jnp.asarray(init_alpha)

    @jax.named_scope("PSinc")
    def __call__(self, x: Array, key: PRNGKeyArray=None,) -> Array:
        r"""**Arguments:**

        - `x`: The input.

        **Returns:**

        A JAX array of the same shape as the input.
        """
        with jax.numpy_dtype_promotion("standard"), jax.numpy_rank_promotion("allow"):
            return jnp.sin(10*x)/jnp.exp(self.damp*x)

def sinc(x):
    return jnp.sin(10*x) / jnp.exp(jnp.abs(0.01 * x))
    
def prox_lasso(params: optax._src.base.Params, lr: float, lreg: float) -> optax._src.base.Params:
    """Taken from https://github.com/google/jaxopt/blob/main/jaxopt/_src/prox.py#L76"""
    if type(l1reg) == float:
        l1reg = jax.tree.map(lambda y: l1reg * jnp.ones_like(y), params)

    def fun(u, lambd):
        return jnp.sign(u) * jax.nn.relu(jnp.abs(u) - lambd * lr)

    return jax.tree_map(fun, params, lreg)

def prox_ridge(params: optax._src.base.Params, lr: float, lreg: float) -> optax._src.base.Params:
    """Taken from https://github.com/google/jaxopt/blob/main/jaxopt/_src/prox.py#L169"""
    if lreg is None:
        lreg = 1.0

    factor = 1. / (1 + lr * lreg)
    return jo.tree_util.tree_scalar_mul(factor, params)

def theta_error(theta_hat, theta, gamma):
  return jnp.nan_to_num(1e-1*jnp.abs(gamma)*(((theta_hat-theta)/theta)**2)*(1+jnp.where(theta_hat/theta<0, 1, 0)*jnp.abs(gamma)))

def linspaced_func(array, x, mini:Int, maxi:Int):
    L = len(array)
    if len(x.shape)<2:
        x = jnp.repeat(x[:,jnp.newaxis], array.shape[1], axis=1)
    ic = x*(L-1)/(maxi-mini)
    binf = ic.astype(jnp.int16)
    bsup = binf+1
    try:
        return (bsup-ic)*array[binf[:,0],:]+(ic-binf)*array[bsup[:,0],:]
    except IndexError:
        if binf[:,0][-1]==L-1:
            return jnp.concatenate(((bsup-ic)[:-1]*array[binf[:,0][:-1],:]+(ic-binf)[:-1]*array[bsup[:,0][:-1],:], 
                                    array[:,binf[:,0][-1]]))
        print(f"x should be smaller than maxi, here {x}>={maxi}")
        raise



class Callable_dict(dict):

    @property
    def __class__(self):
        return PINN
    
    def init_params(self, *args):
        return list(self.values())[0].init_params(*args)
    
    def _eval_nn(self, *args):
        return list(self.values())[0]._eval_nn(*args)

    def __getattr__(self, name):
        try:
            self.__getattribute__(name)
        except:
            try :
                list(self.values())[0].__getattribute__(name)
            except:
                raise AttributeError("AttributeError: attribut non présent dans le dictionnaire ou dans un élément du dictionnaire.")

    def __setitem__(self, key, value):
        for name, values in value.__dict__.items():
            setattr(self, name, values)
        super().__setitem__(key, value)

    def __call__(self, t, params, *args):
        self.coefs=[1,1]
        if isinstance(params["nn_params"], dict):
            return self.coefs[0]*super().__getitem__("1")(t, params["nn_params"]["1"], *args) + self.coefs[1]*super().__getitem__("2")(t, params["nn_params"]["2"], *args)
        else:
            raise Exception("Veuillez utiliser un seul layer dans ce cas.")


def soft_thresholding_additive_update(
    learning_rate: optax.GradientTransformation,
    lreg: float = 1, proximal: Callable = prox_lasso
) -> optax._src.base.GradientTransformation:
    """Soft thresholding operator, given input gradients `grads` return an
    update
    u <- - params + max(0, params - grads - lr * l1reg)

    Parameters
    ----------
    learning_rate : optax.GradientTransformation
        _description_
    l1reg : float, optional
        _description_, by default 1

    Returns
    -------
    base.GradientTransformation
    """

    def init_fn(params: optax._src.base.Params) -> EmptyState:
        del params  # not used in init
        return EmptyState()

    def update_fn(grads, state, params=None):

        # TODO: if learning_rate is a Scheduler (callable) then get the current
        # value of the lr at current iteration
        # see
        # https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py#L773
        # and
        # https://github.com/google-deepmind/optax/blob/main/optax/_src/transform.py#L775

        # Provided `grads`= -lr * \nabla f(x), the following is equivalent to
        #  xnew = x - lr * \nabla f(x)
        gd_update = optax.apply_updates(params, grads)

        # soft thresholding of xnew
        proximal_update = proximal(gd_update, lr=learning_rate, lreg=lreg)

        # we return xold - prox_update to be compatible
        # with optax.apply_update(xold, additive_update) which is additive
        additive_update = jax.tree_map(lambda u,v : v-u, params, proximal_update)

        return additive_update, state

    return optax._src.base.GradientTransformation(init_fn, update_fn)


def proximal_gradient_optax(
    learning_rate: optax._src.base.ScalarOrSchedule, lreg: float = 1.0, proximal: Callable = prox_lasso
) -> optax._src.base.GradientTransformation:
    """Computes proximal gradient updates for l1 regularization.
    The updates are **additive** from the current point, and of the form
    $$
    u_{k+1} <- prox(x_k - lr * \nabla f(x_k)) - x_k,

    prox(u) = max(0, |u| - lr * l1reg) (SoftThresholding)
    $$
    This is order to be compatible with
    optax.apply_updates(x, update)

    x_{k+1} <- x_k + u_{k+1} = prox(x_k - lr * \nabla f(x_k))

    This could introduce roundoff differences compare to a direct
    approach not relying on `optax.apply_updates()`.


    Parameters
    ----------
    learning_rate : base.ScalarOrSchedule
        The learning rate
    l1reg : float, optional
        the regularization, by default 1

    Returns
    -------
    base.GradientTransformation
    """
    vanilla_gd = optax.sgd(learning_rate=learning_rate)

    return optax.chain(
        vanilla_gd,
        soft_thresholding_additive_update(learning_rate=learning_rate, lreg=lreg, proximal=proximal),
    )


class AlternateTxState(NamedTuple):
    step: jnp.ndarray
    tx1_state: Any
    tx2_state: Any

def alternate_tx(tx1, tx2, evry1, evry2):
    def init_fn(params):
        return AlternateTxState(
            step=jnp.zeros([], dtype=jnp.int32),
            tx1_state=tx1.init(params),
            tx2_state=tx2.init(params),
        )
    
    def _update_tx1(updates, state, params=None):
        new_updates, new_state = tx1.update(updates, state.tx1_state, params)
        #jax.lax.cond(state.step%1000==0, lambda _: jax.debug.print("Adam"),lambda _:_,None)
        return new_updates, state._replace(step=state.step+1, tx1_state=new_state)

    def _update_tx2(updates, state, params=None):
        new_updates, new_state = tx2.update(updates, state.tx2_state, params)
        #jax.lax.cond(state.step%1000==0, lambda _: jax.debug.print("ProxGrad"),lambda _:_,None)
        return new_updates, state._replace(step=state.step+1, tx2_state=new_state)

    def update_fn(updates, state, params=None):
        return jax.lax.cond(
            state.step%(evry1+evry2)>=evry1,
            _update_tx2,
            _update_tx1,
            updates, state, params
        )

    return optax.GradientTransformation(init_fn, update_fn)


def optimizer_alternate(list_first_params, list_second_params, tx1, tx2, evry1, evry2) :
    def map_nested_fn(fn):
        """Recursively apply `fn` to the key-value pairs of a nested dict"""
        def map_fn(nested_dict):
            return {
                k: (map_fn(v) if isinstance(v, dict) else fn(k, v))
                for k, v in nested_dict.items()
            }

        return map_fn

    label_fn = partial(map_nested_fn(lambda k, _: k))
    # mask_fn1 = partial(map_nested_fn(lambda k,v : k in list_first_params))
    # mask_fn2 = partial(map_nested_fn(lambda k,v : k in list_second_params))
    return alternate_tx(optax.multi_transform(
        {k: tx1 for k in list_first_params}
        | {k: optax.set_to_zero() for k in list_second_params},  # those gradient transforms must correspond to leaves of parameter pytree
        label_fn),  
        optax.multi_transform({k: optax.set_to_zero() for k in list_first_params}
        | {k: tx2 for k in list_second_params},  # those gradient transforms must correspond to leaves of parameter pytree
        label_fn),
        evry1, evry2)
    # return alternate_tx(optax.masked(tx1, mask_fn1), optax.masked(tx2, mask_fn2),
    #                     evry1, evry2)