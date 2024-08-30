import jax.numpy as jnp


##############

def GLVlogfun(z, p):
    n = z.shape[0]
    GLV = p[:, 0].reshape(-1, 1, order='F') + jnp.dot(p[:, 1:(n+1)], jnp.exp(z.T))
    r = GLV.T
    return jnp.array(r)

     
def GLVlogfunode(y, t, p):
    i = y.shape[0]
    mu = p[:, 0]
    A = p[:, 1:(i+1)]
    r = mu + jnp.dot(A, jnp.exp(y))
    return jnp.array(r)

##############
