import sys
sys.path.append("./version-with-classes/src")
sys.path.append("./version-with-classes/")
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['savefig.directory'] = "~/PINN/Presentation/outputs"
mpl.rcParams['savefig.format'] = "svg"
from jax import random, vmap
import jax.numpy as jnp
import equinox as eqx
import jinns
import pandas as pd
from time import perf_counter as chrono
import scipy.integrate as si
from jinns_glv import Forward_Problem_GLV

key = random.PRNGKey(2)
key, subkey = random.split(key)
batch_size = 16

n = 800
Tmax = 20
epochs = 10
split_data_ODE = 0.1
extension = 0

t = jnp.linspace(0, Tmax*(n+extension)/n, n+extension)

Ns1 = 4
N_01 = jnp.array([.1, .05, .1, .02])
theta1 = jnp.array([[1., -1, -1.09, -1.52, 0], 
                 [.72, 0, -0.72, -0.3168,- 0.9792], 
                 [1.53, -3.5649, 0, -1.53, -0.7191],
                 [1.27, -1.5367, -0.6477, -0.4445, -1.27]])   # Theta pour solution pérdiodique


Ns2 = 20
N_02 = random.uniform(key, shape=(20,))*10
theta2 = jnp.asarray(pd.read_csv("./version-with-classes/data/20_theta.csv", header=None).values)

Ns3 = 5
N_03 = jnp.array([225.399163341	, 0.007395471	, 0.002465157	, 0.007395471	, 0.012325785])
theta3 = jnp.array([[-0.19902241230010986, 0.0019387921784073114, 0.005207688082009554, 0.013450967147946358, -0.0448932871222496, 0.06212534010410309],
 [0.5865827798843384, 0.013972925953567028, -0.004251197446137667, -0.042669739574193954, -0.008555068634450436, 0.14238691329956055],
 [1.7769122123718262, 0.0023382329382002354, 0.03884835168719292, -0.08590973913669586, -0.05712074413895607, -0.7526372671127319],
 [0.8927175998687744, 0.014376112259924412, 0.0029520844109356403, -0.05564774572849274, -0.039456628262996674, 0.08949698507785797],
 [0.5857483148574829, 0.005975721869617701, -0.007333327084779739, -0.030138956382870674, 0.001018781098537147, 0.16007456183433533]])


Ns = Ns2
N_0=N_02
theta = theta2


def f(y, t, theta, Tmax):
  return (theta[:,[0]] + theta[:,1:]@jnp.exp(y).reshape((theta.shape[0], 1))).ravel()

ode_chrono = chrono()
solutions = jnp.array(si.odeint(f, jnp.log(N_0), t, args=(theta, Tmax,)))
ode_chrono = chrono()-ode_chrono
print("ode time:", ode_chrono)
max_sol = jnp.max(solutions)

subsample = jnp.linspace(0, n, batch_size).astype(jnp.int16)
solutions_approx = jnp.log(jnp.abs(jnp.exp(solutions[subsample]) + max_sol / 20 * random.generalized_normal(subkey, 2, shape=(batch_size, Ns))))  # adds noise to data (poor implementation)
theta_approx = theta + jnp.max(theta)/5 * random.generalized_normal(subkey, 2, shape=theta.shape)
data = {0:(t[subsample]/Tmax)[:,jnp.newaxis], 1:solutions_approx}

problem = Forward_Problem_GLV(theta, Ns, n, batch_size, Tmax, N_0=jnp.log(N_0), layers=1)

begin = chrono()
lambda_eq = 0
lambda_ini = 1
lambda_data = 1
lambda_tot=(lambda_eq+lambda_ini+lambda_data)/3
(loss, loss_dict, _) = problem.evaluate(int(split_data_ODE * epochs), lambda_eq / lambda_tot, lambda_ini / lambda_tot, lambda_data / lambda_tot, data=data)
time = chrono()-begin
lambda_eq = 0.1
lambda_ini = 1
lambda_data = 1
lambda_tot=(lambda_eq+lambda_ini+lambda_data)/3
(loss2, loss_dict2, validation_loss) = problem.evaluate(int((1-split_data_ODE)*epochs), lambda_eq/lambda_tot, lambda_ini/lambda_tot, lambda_data/lambda_tot, data=data, mode="validation")

stop_i = jnp.count_nonzero(loss2)
loss2 = loss2[:stop_i]
loss_dict2 = { key:value[:stop_i] for key,value in loss_dict2.items() }

loss=jnp.concatenate((loss, loss2))

u_est_fp = []
u_est_fp_sin = []
for i in range(Ns) :
  u_est_fp.append(vmap(lambda t:jnp.exp(problem.pinn.forward(t)[i])))
# for i in range(Ns):
#   u_est_fp_sin.append(vmap(lambda t:jnp.exp(problem.pinn.sec_u(t, problem.pinn.params["nn_params"]["2"])[i])))


key, subkey = random.split(key, 2)
val_data = jinns.data.DataGeneratorODE(subkey, n, 0, 1, batch_size, 'uniform').times.sort(axis=0)
val_data=jnp.concatenate((jnp.array([0]), val_data))
plt.figure("Resultats.svg")
for i in range(Ns) :
  #plt.plot(t, u_est_fp[i](t / Tmax), '-' , label="N" + str(i)+" Total", color='C'+str(i))
  # plt.plot(t, u_est_fp_sin[i](t / Tmax), '-' , label="N" + str(i) + " Sinc seulement")
  plt.plot(t, jnp.exp(solutions[:,i]), '-', color='C'+str(i))
  #plt.plot(t[subsample], jnp.exp(solutions_approx)[:,i], 'x', color='C'+str(i))

plt.legend()
plt.figure("Loss.svg")
for loss_name, loss_values in loss_dict.items():
    plt.plot(jnp.log10(jnp.concatenate((loss_values,loss_dict2[loss_name]))), label=loss_name)

print(loss.shape)
print(f"total loss: {loss[-1]}")
print(f"Individual losses: { {key: f'{val[-1]:.5f}' for key, val in loss_dict2.items()} }")
print("Time spent in training : ", time)
plt.plot(jnp.log10(loss), label="total loss")
plt.plot(range(int(epochs*split_data_ODE), epochs), jnp.log10(validation_loss), label = "Validation loss")
plt.legend()
plt.show()