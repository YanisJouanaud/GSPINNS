from fun_examples.GLV_functions import *
from GSA_fun import *

# Set parameters
norder = 3
fold = 2
lam = 10 ** 2.5
lam2 = 10 ** -2.5
lam0 = 0.01
iterationMax = 100
errMax = 1e-5
use_jax = True 
verbose = 0

# Read the data
filename = 'data_examples/database.txt'
allexp = np.genfromtxt(filename, delimiter=',')

# Specify the species (columns) you want to work with
indcol = range(2, allexp.shape[1])
speciesexp = allexp[:, indcol]
n = speciesexp.shape[1]

# Specify species names for the plots
namespecies = ['group 1', 'group 2', 'group 3', 'group 4']

# Specify scaling for each species
scale = 10**0 * np.ones(len(namespecies))

## Define the GLV model functions
fn = {'fn': GLVlogfun}

# Data processing
ind_exp = np.unique(allexp[:, 0])
n_exp = len(ind_exp)

temps = np.sort(np.unique(allexp[:, 1]))
rg = [np.min(temps), np.max(temps)]
tini = temps[0]
horizT = temps[-1] - tini
tspan = np.arange(tini, tini + horizT + 1)
nknots = fold * (len(tspan) - 1) + 1
#
## Create cell arrays for data storing
Tcell = [[[]]*n]*n_exp
path_cell = [[[]]*n]*n_exp
Ycell = [[[]]*n]*n_exp

for j in range(n_exp):
  ind = np.where(allexp[:, 0] == ind_exp[j])[0]
  tt = allexp[ind, 1]
  tini = tt[0]
  species = speciesexp[ind, :]

  for i in range(n):
    obs_pts = np.where(species[:, i] >= 0)[0]
    Tcell[j][i] = tt[obs_pts]
    path_cell[j][i] = species[obs_pts, i]
    Ycell[j][i] = np.log(np.maximum(path_cell[j][i] / scale[i], 1e-8))

Tcell = np.array(Tcell)
path_cell = np.array(path_cell)
Ycell = np.array(Ycell)

## Compute weights for the data
wts = np.ones_like(Ycell)

# Set smoothing weight parameters
wts_lambda = np.ones(n)
lam = lam * wts_lambda

# Set optimization parameters for least squares solver
#lsopts_c = {'verbose':1, 'ftol': 1e-15, 'xtol': 1e-14}
lsopts_c = {'verbose':verbose}


## Run the alternate minimization
start_time = time.time()
newp, newcoefs = Alternate_multi(fn, wts, lam, lam2, lam0, iterationMax, errMax, n, Tcell, Ycell, norder, nknots, rg, lsopts_c, use_jax, verbose)
elapsed_time = time.time() - start_time
print('--------------------------------------')
print("Elapsed time:", elapsed_time, "seconds")
##
## Display estimated parameters
print('New parameter estimates:')
print(newp)

np.save('results/param', newp)
np.save('results/coefs', newcoefs)

plot_results(nknots, rg, norder, n, newcoefs, Tcell, Ycell, newp, namespecies, GLVlogfunode)


