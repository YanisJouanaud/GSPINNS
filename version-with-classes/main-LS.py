import sys
sys.path.append("./version-with-classes/src")
sys.path.append("./version-with-classes/")
import argparse
from functions import *
import time
from glv import GLVlogfun, GLVlogdfdx, GLVlogfunode

algo = GeneralisedSmoothingSpline()

# Set parameters
algo.errMax =  1e-3
algo.iterationMax = 100

# Read the data
filename = './version-with-classes/data/data.csv'
algo.load_data(filename)

## Define the model 
odefn = GLVlogfunode
fn = {'fn': GLVlogfun, 'dfdx': GLVlogdfdx}

## Run the algorithm
start_time = time.time()
print('--------------------------------------')
print('Running alternate minimization with least square...')
estimated_parameters, newcoefs = algo.run_alternate_multi(fn, verbose=0, jac_analytical=False)
elapsed_time = time.time() - start_time
print("Elapsed time:", elapsed_time, "seconds")

print('Estimated parameters:\n', estimated_parameters)

## Plot the results
algo.plot_results(estimated_parameters, newcoefs, odefn)

#if save:
#    export_result(estimated_parameters, save_path, nn=False)
