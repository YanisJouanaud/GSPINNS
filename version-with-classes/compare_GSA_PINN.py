import sys, os
sys.path.append("./version-with-classes/src")
sys.path.append("./version-with-classes/")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.directory'] = "~/PINN/gsa_algorithm/outputs"
mpl.rcParams['savefig.format'] = "svg"
from jax import random, vmap, config
import jax.numpy as jnp
import pandas as pd
from time import perf_counter as chrono
import scipy.integrate as si
from src.functions import *
from src.glv import GLVlogfun, GLVlogdfdx, GLVlogfunode
from src.jinns_glv import GeneralisedSmoothingJINN
from src.utils import theta_error

# config.update("jax_enable_x64", True)
# config.update("jax_debug_nans", True)

key = random.PRNGKey(2)
key, subkey = random.split(key)

##### HYPERPARAMETERS ######

n = 320
epochs = 500
extension = 0
n_iter = 100
epochs_init = 25000

prox_coef = 0.1
lmbda = 2

############################


def f(y, t, theta, Tmax):
  return (theta[:,[0]] + theta[:,1:]@jnp.exp(y).reshape((theta.shape[0], 1))).ravel()

###### DATA CREATION ######

# batch_size = 16
# Tmax = 10
# t = jnp.linspace(0, Tmax*(n+extension)/n, n+extension)

# Ns1 = 3
# N_01 = jnp.array([5., 3., 1.])
# theta1 = jnp.array([[7.5,-2,-5,-.5],
#                     [2.6,-.5,-1,-1.2],
#                     [2.5, -1, -.5, -1]])  # Theta pour solution périodique

# Ns2 = 20
# N_02 = random.uniform(key, shape=(20,))*10
# theta2 = jnp.asarray(pd.read_csv("./version-with-classes/data/20_theta.csv", header=None).values)

# Ns = Ns1
# N_0 = N_01
# theta = theta1
          


# solutions = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta, Tmax,)))

# max_sol = jnp.max(solutions)

# subsample = jnp.linspace(0, n, batch_size).astype(jnp.int16)
# solutions_approx = jnp.log(jnp.abs(jnp.exp(solutions[subsample]) + max_sol / 20 * random.generalized_normal(subkey, 2, shape=(batch_size, Ns))))  # adds noise to data (poor implementation)
# data = {0:t[subsample], 1:solutions_approx}
###########################


##### REAL DATA #####

rdata = pd.read_csv("./version-with-classes/data/fromage.csv")
data_plot = rdata
# rdata["time"] = rdata["t"] + 0.01*(rdata["#Exp"]-1)
# rdata.drop(["t", "#Exp"], axis=1, inplace=True)
# rdata = rdata.set_index("time").sort_index()
rdata = rdata[rdata["#Exp"]==1].set_index("t").sort_index().drop("#Exp", axis=1)
rdata = rdata.where(rdata != 0).dropna(axis=1, how='any')
rdata *= 1e3/rdata.max(axis=None)
print(rdata)
time_data = np.asarray(rdata.index, dtype=np.float32)
time_data -= time_data[0]
batch_size = len(time_data)
sp_data = jnp.asarray(rdata.values)
data = {0:time_data, 1:jnp.log(sp_data)}

N_0 = sp_data[0,:]
Tmax = time_data[-1]
Ns = len(N_0)
t = jnp.linspace(0, Tmax*(n+extension)/n, n+extension)
######################

###### PINN RESOLUTION ######
timer_PINN = chrono()
algorithm = GeneralisedSmoothingJINN(Ns, Tmax, N_0, n, batch_size, key, 
                                     lambd0=lmbda, lambd1=0, lambd2=1,
                                     iterationMax = n_iter, errMax = 1e-1)

print("Running PINN algorithm...")
params, err = algorithm.run_alternate(epochs_init, epochs,
                                      data, prox_coef=prox_coef,
                                      verbose=False)
timer_PINN = chrono() - timer_PINN
##############################


##### GSA RESOLUTION #####

timer_GSA = chrono()
algo = GeneralisedSmoothingSpline()

# Set parameters
algo.errMax = 1e-4
algo.iterationMax = 100
algo.nb_species = Ns
algo.namespecies = [f'N{i+1}' for i in range(Ns)]
algo.scale = jnp.ones(len(algo.namespecies))
algo.n_exp = 1

algo.Tcell = jnp.reshape(jnp.repeat(data[0], Ns, axis=0), (1,batch_size,Ns)).transpose(0,2,1)
algo.path_cell = jnp.reshape(jnp.exp(data[1]), (1,batch_size,Ns)).transpose(0,2,1)
algo.Ycell = jnp.reshape(data[1], (1,batch_size,Ns)).transpose(0,2,1)
temps = data[0]
tini = temps[0]
horizT = temps[-1] - tini
tspan = jnp.arange(tini, tini + horizT + 1)
algo.rg = [jnp.min(temps), jnp.max(temps)]
algo.nknots = algo.fold * (len(tspan) - 1) + 1
algo.nbasis = algo.nknots + algo.norder - 2
odefn = GLVlogfunode
fn = {'fn': GLVlogfun, 'dfdx': GLVlogdfdx}
print("Running GSA algorithm...")
estimated_parameters, newcoefs = algo.run_alternate_multi(fn, verbose=0, jac_analytical=False)
timer_GSA = chrono() - timer_GSA

##########################


##### COMPARISON #####

print("Time for PINN :", timer_PINN, ", Time for GSA :", timer_GSA)
u_est_fp = []
for i in range(Ns) :
  u_est_fp.append(vmap(lambda t:jnp.exp(algorithm.u.forward(t, params)[i])))
vecnorm_u = jnp.linalg.norm(jnp.array([u_est_fp[i](t/Tmax) for i in range(Ns)]), axis=1)
theta_found = params["eq_params"]["theta"]


theta_known = False

if theta_known:
  error_mu_PINN = theta_error(theta_found[:,[0]], theta[:,[0]], vecnorm_u[:, jnp.newaxis]*theta[:,[0]]/jnp.linalg.norm(theta))
  error_A_PINN = theta_error(theta_found[:,1:], theta[:,1:], vecnorm_u*theta[:,1:]/jnp.linalg.norm(theta))

  print(jnp.count_nonzero(jnp.where(theta_found/theta<=0, 1, 0))/jnp.size(theta))
  error_theta_PINN =  jnp.linalg.norm(jnp.concatenate((error_mu_PINN, error_A_PINN), axis=1))

  theta_found = estimated_parameters

  error_mu_GSA = theta_error(theta_found[:,[0]], theta[:,[0]], vecnorm_u[:, jnp.newaxis]*theta[:,[0]]/jnp.linalg.norm(theta))
  error_A_GSA = theta_error(theta_found[:,1:], theta[:,1:], vecnorm_u*theta[:,1:]/jnp.linalg.norm(theta))

  error_theta_GSA = jnp.linalg.norm(jnp.concatenate((error_mu_GSA, error_A_GSA), axis=1))

  print("Error with PINN :", error_theta_PINN , ",\nError with GSA :", error_theta_GSA)


algorithm.u.save_pinn("outputs/PINN")
jnp.save("outputs/Theta_found", params["eq_params"]["theta"])

solutions_found_PINN = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta_found, Tmax,)))
solutions_found_GSA = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(estimated_parameters, Tmax,)))

ew_error_PINN = (jnp.log(rdata.values)-solutions_found_PINN[jnp.array(time_data/batch_size*n, dtype=jnp.int16)])**2
error_PINN = 1/jnp.size(ew_error_PINN)*jnp.sum(ew_error_PINN)

ew_error_GSA = (jnp.log(rdata.values)-solutions_found_GSA[jnp.array(time_data/batch_size*n, dtype=jnp.int16)])**2
error_GSA = 1/jnp.size(ew_error_GSA)*jnp.sum(ew_error_GSA)

print("Error with PINN :", error_PINN , ",\nError with GSA :", error_GSA)
plt.figure()
plt.yscale("log")
for i in range(Ns) :
  plt.plot(t+rdata.index[0],jnp.exp(solutions_found_PINN[:,i]), '-' , label="PINN Solution N" + str(i), color='C'+str(i))
  plt.plot(t+rdata.index[0], jnp.exp(solutions_found_GSA[:,i]), '-.', color='C'+str(i), label="GSA Solution N"+str(i))
plt.plot(rdata, "x")
plt.legend()
plt.show()
plt.savefig("outputs/Resultats.svg")