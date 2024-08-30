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
from jinns_glv import Inverse_Problem_GLV
from utils import theta_error

key = random.PRNGKey(2)
key, subkey = random.split(key)

########################
batch_size = 16
n = 320   # number of points for GLV-loss
Tmax = 10  # End time
n_iter = 70  # nombre d'itération de training
extension = 0  # number of additional points at the right of the graph
lambda1, lambda2, lambda3 = 0.1, 1, 1  # coefficient for the loss repartition between GLV, initial and data loss
epochs_init = 6000
prox_coef = 1e-2
################################



t = jnp.linspace(0, Tmax*(n+extension)/n, n+extension)

Ns1 = 3
N_01 = jnp.array([5., 3., 1.])
theta1 = jnp.array([[7.5,-2,-5,-.5],
                    [2.6,-.5,-1,-1.2],
                    [2.5,-1,-.5,-1]])   # Theta pour solution périodique


Ns2 = 20
N_02 = random.uniform(key, shape=(20,))*10
theta2 = jnp.asarray(pd.read_csv("./version-with-classes/data/20_theta.csv", header=None).values)

Ns = Ns1
N_0=N_01
theta = theta1


def f(y, t, theta, Tmax):
  return (theta[:,[0]] + theta[:,1:]@jnp.exp(y).reshape((theta.shape[0], 1))).ravel()
solutions = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta, Tmax,)))


max_sol = jnp.max(solutions)

subsample = jnp.linspace(0, n, batch_size).astype(jnp.int16)
solutions_approx = jnp.log(jnp.abs(jnp.exp(solutions[subsample]) + 
                                   max_sol / 20 * 
                                   random.generalized_normal(subkey, 2, shape=(batch_size, Ns))))  # adds noise to data (poor implementation)

random_theta = random.ball(subkey, 1, 2, (Ns, Ns+1))[:,:,0]
data = [t[subsample], solutions_approx]

timer = chrono()
algorithm = Inverse_Problem_GLV(random_theta, Ns, n, batch_size, Tmax, jnp.log(N_0))
(loss, loss_by_term, 
 other, theta_found, layers) = algorithm.run(epochs_init, 
                                            n_iter, data, lambda1, 
                                            lambda2, lambda3,
                                            prox_coef, (700,10000))
timer = chrono() - timer

plt.figure("Resultats.svg")
u_est_fp = []
for i in range(Ns) :
  u_est_fp.append(vmap(lambda t:jnp.exp(algorithm.pinn.forward(t)[i])))
for i in range(Ns) :
  plt.plot(t, u_est_fp[i](t / Tmax), '-' , label="N" + str(i)+" Total", color='C'+str(i))
  # plt.plot(t, u_est_fp_sin[i](t / Tmax), '-' , label="N" + str(i) + " Sinc seulement")
  plt.plot(t, jnp.exp(solutions[:,i]), '--', color='C'+str(i))
  plt.plot(t[subsample], jnp.exp(solutions_approx)[:, i], 'x', color='C' + str(i))

plt.legend()
jnp.save("outputs/loss", loss)
jnp.save("outputs/loss_by_term", loss_by_term)
jnp.save("outputs/Theta_found", theta_found)
jinns.utils.save_pinn("outputs/PINN", algorithm.pinn.u, algorithm.pinn.params,
                      {"eqx_list":layers, "type": "ODE"})
print(theta/theta_found)
print(theta_found)
vecnorm_u = jnp.linalg.norm(jnp.array([u_est_fp[i](t/Tmax) for i in range(Ns)]), axis=1)
error_mu = theta_error(theta_found[:,[0]], theta[:,[0]], 
                       vecnorm_u[:, jnp.newaxis]*theta[:,[0]]/jnp.linalg.norm(theta))
error_A = theta_error(theta_found[:,1:], theta[:,1:], 
                      vecnorm_u*theta[:,1:]/jnp.linalg.norm(theta))
print(jnp.linalg.norm(jnp.concatenate((error_mu, error_A), axis=1)))
print(timer)
plt.savefig("outputs/Resultats.svg")