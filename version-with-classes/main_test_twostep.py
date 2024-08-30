import sys
sys.path.append("./version-with-classes/src")
sys.path.append("./version-with-classes/")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.directory'] = "~/PINN/gsa_algorithm/outputs"
mpl.rcParams['savefig.format'] = "svg"
from jax import random, vmap
import jax.numpy as jnp
import equinox as eqx
import jinns
import pandas as pd
from time import perf_counter as chrono
import scipy.integrate as si
from jinns_glv import GeneralisedSmoothingJINN
from utils import theta_error

key = random.PRNGKey(2)
key, subkey = random.split(key)
##### HYPERPARAMETERS ######
batch_size = 16
n = 320
Tmax = 10
epochs = 1000
extension = 0
n_iter = 5
epochs_init = 2000

############################

t = jnp.linspace(0, Tmax*(n+extension)/n, n+extension)

Ns1 = 3
N_01 = jnp.array([5., 3., 1.])
theta1 = jnp.array([[7.5,-2,-5,-.5],
                  [2.6,-.5,-1,-1.2],
                  [2.5,-1,-.5,-1]])   # Theta pour solution périodique


Ns2 = 20
N_02 = random.uniform(key, shape=(20,))*10
theta2 = jnp.asarray(pd.read_csv("./version-with-classes/data/20_theta.csv", header=None).values)

Ns = Ns2
N_0 = N_02
theta = theta2


def f(y, t, theta, Tmax):
  return (theta[:,[0]] + theta[:,1:]@jnp.exp(y).reshape((theta.shape[0], 1))).ravel()

solutions = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta, Tmax,)))

max_sol = jnp.max(solutions)

subsample = jnp.linspace(0, n, batch_size).astype(jnp.int16)
solutions_approx = jnp.log(jnp.abs(jnp.exp(solutions[subsample]) + max_sol / 20 * random.generalized_normal(subkey, 2, shape=(batch_size, Ns))))  # adds noise to data (poor implementation)

data = [t[subsample], solutions_approx]

timer = chrono()
algorithm = GeneralisedSmoothingJINN(Ns, Tmax, N_0, n, batch_size, key,
                                     lambd0=0.1, lambd1=1, lambd2=1, 
                                     iterationMax = n_iter, errMax = 1e-4)
params, err = algorithm.run_alternate(epochs_init, epochs, data, prox_coef=1e0, verbose=True)
theta_found = params["eq_params"]["theta"]
timer -= chrono()
print(theta_found/theta)

solutions_found = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta_found, Tmax,)))

plt.figure("PINN")


u_est_fp = []
for i in range(Ns) :
  u_est_fp.append(vmap(lambda t:jnp.exp(algorithm.u.forward(t, params)[i])))
for i in range(Ns) :
  plt.plot(t, u_est_fp[i](t / Tmax), '-' , label="N" + str(i)+" Total", color='C'+str(i))
  plt.plot(t, jnp.exp(solutions[:,i]), '--', color='C'+str(i))
  plt.plot(t[subsample], jnp.exp(solutions_approx)[:, i], 'x', color='C' + str(i))

plt.legend()
plt.figure("Theta")
for i in range(Ns) :
  plt.plot(t, jnp.exp(solutions[:,i]), '--', color='C'+str(i))
  plt.plot(t, jnp.exp(solutions_found[:,i]), '-', color='C'+str(i))
plt.legend()



vecnorm_u = jnp.linalg.norm(jnp.array([u_est_fp[i](t/Tmax) for i in range(Ns)]), axis=1)
error_mu = theta_error(theta_found[:,[0]], theta[:,[0]], vecnorm_u[:, jnp.newaxis]*theta[:,[0]]/jnp.linalg.norm(theta))
error_A = theta_error(theta_found[:,1:], theta[:,1:], vecnorm_u*theta[:,1:]/jnp.linalg.norm(theta))

print("Theta-error :", jnp.linalg.norm(jnp.concatenate((error_mu, error_A), axis=1)))
print("ratio of wrong sign :", jnp.count_nonzero(jnp.where(jnp.logical_and(theta_found/theta<0, jnp.abs(theta-theta_found)>jnp.max(theta)/100),1, 0))/jnp.size(theta))

print("time :", -timer)
plt.show()