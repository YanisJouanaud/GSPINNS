import sys, os
sys.path.append("./version-with-classes/src")
sys.path.append("./version-with-classes/")
from src.jinns_glv import *
import jinns
import jax.random as jrandom
import jax.numpy as jnp
from scipy.integrate import odeint
import scipy.stats as st
from src.utils import theta_error as te
from src.utils import linspaced_func

jax.config.update("jax_debug_nans", True)

class Stat_Analyzer:
    
    def __init__(self, f, Tmax, data, Nf, key, N0=None, lmbda=1, 
                 prox_coef=1e-1, itermax=100):
        self.f = f
        self.Tmax = Tmax
        self.data = data
        self.Nf = Nf
        self.time = jnp.linspace(0, self.Tmax, self.Nf)
        self.Ns = data[1].shape[1]
        if N0 is None:
            self.N0 = jnp.exp(data[1][0,:])
        else:
            self.N0 = N0
        self.pinn = None

        batch_size = data[0].shape[0]
        self.algorithm = GeneralisedSmoothingJINN(self.Ns, self.Tmax, jnp.exp(data[1][0,:]), 
                                             self.Nf, batch_size, key, 
                                             lambd0=lmbda, lambd1=0, lambd2=1,
                                             iterationMax=itermax, errMax=1e-2)
        params, err = self._evaluate_theta(lmbda, 1, prox_coef)
        self.theta = params["eq_params"]["theta"]

    def _evaluate_theta(self, lmbda, weight, prox_coef,
                        epochs_init=25000, epochs=500, data=None):
        
        if data is None:
            data = self.data
        
        self.algorithm.set_N0(jnp.exp(data[1][0,:]))
        self.algorithm.set_lam0(lmbda)
        self.algorithm.set_lam2(weight)
        params, err = self.algorithm.run_alternate(epochs_init, epochs,
                                             data, prox_coef=prox_coef,
                                             verbose=False)
        self.pinn = lambda t: self.algorithm.u(t, params)
        return params, err

    def sliding_prediction_error(self, K_delta:int, nh:int, theta=None):
        if theta is None:
            theta = self.theta
        delta = 1/K_delta
        SPE = 0
        for j in range(K_delta-nh):
            t = jnp.linspace(j*delta, delta*(j+nh), self.Nf)
            x = jnp.array(odeint(self.f, 
                                 self.pinn(delta*j),
                                  t, args=(theta, self.Tmax,)))
            x_data = linspaced_func(x, self.data[0], j*delta, delta*(j+nh))
            SPE += jnp.sum((self.data[1]-x_data)**2)
        return SPE/(K_delta-nh+1)
            

    def theta_error(self, rtheta):
        vecnorm_u = jnp.linalg.norm(jnp.array([self.pinn[i](self.time/self.Tmax) 
                                               for i in range(self.Ns)]), axis=1)
        error_mu = te(self.theta[:,[0]], rtheta[:,[0]], 
                      vecnorm_u[:, jnp.newaxis]*rtheta[:,[0]]/jnp.linalg.norm(rtheta))
        error_A = te(self.theta[:,1:], rtheta[:,1:], 
                     vecnorm_u*rtheta[:,1:]/jnp.linalg.norm(rtheta))
        return jnp.concatenate((error_mu, error_A), axis=1)


    def bootstrap_estimator(self, lmbda, prox_coef, key, B, n_iter=100, 
                            epochs_init=25000, epochs=500, q=20, data=None):
        if data is None:
            data = self.data
        batch_size = data[1].shape[0]
        key, subkey = jrandom.split(key, 2)
        thetas=[]
        for i in range(B):
            bootstrap_weights = jnp.repeat(jrandom.exponential(key, (1, batch_size)), self.Ns, axis=0).T
            bootstrap_weights /= jnp.sum(bootstrap_weights[0])
            params, err = self._evaluate_theta(lmbda, bootstrap_weights, prox_coef, epochs_init, epochs, data)
            
            theta_found = params["eq_params"]["theta"]
            thetas.append(theta_found)
        thetas = jnp.array(thetas)
        quantiles = jnp.quantile(thetas, jnp.array([i/q for i in range(q)]), axis=0)
        CI = st.t.interval(confidence=0.95, 
                           df=len(thetas)-1, 
                           loc=jnp.mean(thetas, axis=0),
                           scale=st.sem(thetas, axis=0))
        return quantiles, CI


    def hyperparameter_select(self, lambda_grid, eta_grid, 
                              K_delta, nh, key, method='linear', data=None):
        hps = {}
        if method == 'linear':
            for l in lambda_grid:
                for e in eta_grid:
                    key, subkey = jrandom.split(key, 2)
                    self.algorithm.init_params(subkey)
                    params, err = self._evaluate_theta(float(l), 1, float(e), data=data)
                    if not(jnp.isnan(params["eq_params"]["theta"]).any()):
                        SPE = self.sliding_prediction_error(K_delta, nh, 
                                                            theta=params["eq_params"]["theta"])
                        hps[(float(l),float(e))] = SPE
            if hps=={}:
                print("Invalid grid: only nan solutions found")
                raise ValueError
            return min(hps, key=hps.get)
    

    def MC_estimation(self, nb_sim, batch_size, lambda_grid, 
                      eta_grid, K_delta, nh, key):
        
        CP = jnp.zeros_like(self.theta)
        PRS = jnp.zeros_like(self.theta)
        PZL = jnp.zeros_like(self.theta)

        list_params = []

        for i in range(nb_sim):
            key, subkey = jrandom.split(key, 2)

            r_theta = self.theta * jrandom.lognormal(key, 0.1, self.theta.shape)
            r_N0 = jnp.abs(self.N0 * jrandom.lognormal(subkey, 0.1, self.N0.shape))
            key, subkey = jrandom.split(key, 2)
            print(i, r_theta, r_N0)
            x = jnp.array(odeint(self.f, jnp.log(r_N0),
                                 self.time, args=(r_theta, self.Tmax,)))
            subsample = jnp.linspace(0, self.Nf, batch_size).astype(jnp.int16)
            data = [self.time[subsample], x[subsample]]

            lmbda, eta = self.hyperparameter_select(lambda_grid, eta_grid, K_delta, nh, subkey, data=data)
            params, err = self._evaluate_theta(lmbda, 1, eta, data=data)
            quantiles, CI = self.bootstrap_estimator(lmbda, eta, key, 20, data=data)

            CP += jnp.where(jnp.logical_and(CI[0]<=self.theta,CI[1]>=self.theta), 1, 0)
            PRS += jnp.where(self.theta*params["eq_params"]["theta"]>=0, 1, 0)
            PZL += jnp.where(params==0, 1, 0)

            list_params.append(params)
        
        list_params = jnp.array(list_params)

        bias = jnp.mean(jnp.mean(list_params, axis=0)-self.theta)
        MSE = st.sem(list_params, axis=0)
        return CP/nb_sim, PRS/nb_sim, PZL/nb_sim, bias, MSE
    

if __name__=="__main__":

    key = jrandom.PRNGKey(123456789)
    key, subkey = jrandom.split(key)
    batch_size = 16

    n = 320
    Tmax = 7

    Ns = 3
    N_0 = jnp.array([5., 3., 1.])
    theta = jnp.array([[7.5,-2,-5,-.5],
                       [2.6,-.5,-1,-1.2],
                       [2.5,-1,-.5,-1]])   # Theta pour solution pérdiodique
    t = jnp.linspace(0, Tmax, n)

    def f(y, t, theta, Tmax):
        return (theta[:,[0]] + theta[:,1:]@jnp.exp(y).reshape((theta.shape[0], 1))).ravel()
    solutions = jnp.array(odeint(f, jnp.log(N_0), t, args=(theta, Tmax,)))


    max_sol = jnp.max(solutions)

    subsample = jnp.linspace(0, n, batch_size).astype(jnp.int16)
    solutions_approx = jnp.log(jnp.abs(jnp.exp(solutions[subsample]) + max_sol / 20 * jrandom.generalized_normal(subkey, 2, shape=(batch_size, Ns))))  # adds noise to data (poor implementation)
    data = {0:t[subsample], 1:solutions_approx}

    key, subkey = jrandom.split(key)
    experiment = Stat_Analyzer(f, Tmax, data, n, subkey)
    grid = 10.**jnp.linspace(-2, 2, 4)
    (CP, PRS, PZL, bias, MSE )= experiment.MC_estimation(10, batch_size, grid, grid, 128, 32, key)
    print(CP,PRS,PZL,bias,MSE,experiment.theta,theta)
    