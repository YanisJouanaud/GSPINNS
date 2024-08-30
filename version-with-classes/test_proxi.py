from re import X
import sys
sys.path.append("./version-with-classes/src")
sys.path.append("./version-with-classes/")
import matplotlib.pyplot as plt
from jax import random, vmap, jacfwd
import jax.numpy as jnp
import pandas as pd
import scipy.integrate as si
import jaxopt as jo
from utils import theta_error

key = random.PRNGKey(2)
key, subkey = random.split(key)

Tmax = 15

t = jnp.linspace(0,Tmax,500)

Ns = 20
N_0 = random.uniform(key, shape=(20,))*10
theta = jnp.asarray(pd.read_csv("./version-with-classes/data/20_theta.csv", header=None).values)


############ PARAMETERS ############
C_prox = 1e-1
max_iter = 10000
stepsize = 0
prox_op = jo.prox.prox_ridge
####################################


def f(y, t, theta, Tmax):
  return (theta[:,[0]] + theta[:,1:]@jnp.exp(y).reshape((theta.shape[0], 1))).ravel()

solutions = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta, Tmax,)))

def cost_function(theta, data):
  # theta in the form of a (Ns,(Ns+1)) jnp array
  # data in the form of a tuple of 2 jnp (Ns, Nf) arrays and 1 float
  df, expf, tmax = data
  Ns, Nf = df.shape
  mu = theta[:,[0]]
  A = theta[:,1:]
  sum_norm = 1/Nf*jnp.sum(jnp.linalg.norm(df-tmax*(mu+A@expf), axis=0)**2)
  return sum_norm

df = vmap(lambda i:Tmax*f(solutions[i], t, theta, Tmax))(jnp.arange(500)).T
expf = jnp.exp(solutions).T
pg = jo.ProximalGradient(cost_function, prox_op,
                                    stepsize=stepsize, maxiter=max_iter, tol=1e-3, jit=True, verbose=True)
thetabase = random.ball(key, 1, 2, (Ns,Ns+1))[:,:,0]
theta_found = pg.run(thetabase, hyperparams_prox=C_prox, data=(df, expf, Tmax)).params

vecnorm_u = jnp.linalg.norm(expf, axis=1)
error_mu = theta_error(theta_found[:,[0]], theta[:,[0]], vecnorm_u[:, jnp.newaxis]*theta[:,[0]]/jnp.linalg.norm(theta))
error_A = theta_error(theta_found[:,1:], theta[:,1:], vecnorm_u*theta[:,1:]/jnp.linalg.norm(theta))

print(jnp.linalg.norm(jnp.concatenate((error_mu, error_A), axis=1)))

plt.figure()
err = plt.imshow(jnp.abs((theta - theta_found)/theta))
plt.colorbar(err)
# for i in range(Ns):
#    for j in range(Ns+1):
#       plt.text(i,j,str(theta[i,j]), size='xx-small')


print(jnp.count_nonzero(theta)-jnp.count_nonzero(theta_found))
solutions_found = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta_found, Tmax,)))
plt.legend()
plt.figure("Theta")
for i in range(Ns) :
  plt.plot(t, jnp.exp(solutions[:,i]), '--', color='C'+str(i))
  plt.plot(t, jnp.exp(solutions_found[:,i]), '-', color='C'+str(i))
plt.legend()


plt.show()