import jinns
from jinns.solver._solve import solve
from jinns.validation import ValidationLoss
from jinns.utils._pinn import PINN
import jax
import jaxopt as jo
from jax import random, vmap
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax
import matplotlib.pyplot as plt
from utils import *
from time import perf_counter as chrono
from solve_alternate import solve_alternate, Optimizers
from containers import *

class DeChamps_loss:
    def __init__(self):
        pass
    
    def evaluate(self):
        pass


class GLV_loss:
  # Class containing the loss function for the Generalized-Lotka Volterra equation
  def __init__(self, Tmax, J3=0.):
    self.Tmax = Tmax
    self.J3=J3  # whether to add a sparsity penalty

  def evaluate(self, t, u, params):
    theta = params["eq_params"]["theta"]
    udt = jax.jacfwd(lambda x:u(x, params['nn_params']))(t)[:,jnp.newaxis]
    f = udt-self.Tmax*(theta[:,[0]] +   # right-hand side of the ODE
            (theta[:, 1:] @ jnp.exp(u(t, params['nn_params']))[:, jnp.newaxis]))

    k = jnp.sum((params['eq_params']["theta"])**2)/jnp.size(params['eq_params']["theta"])  # penalization for sparsity of parameter
    return f + self.J3*k  # Loss associated to the ODE for input t in PINN u with parameter params

class GLV_loss_duo:
  # Class containing the loss function for the Generalized-Lotka Volterra equation
  # in the case where the neural network is a dictionary of smaller NNs
  def __init__(self, Tmax, J3=0.):
    self.Tmax = Tmax
    self.J3=J3  # whether to add a sparsity penalty

  def evaluate(self, t, u_dict, params):
    theta = params["eq_params"]["theta"]
    u1, u2 = u_dict["1"], u_dict["2"]
    udt = jnp.array([[jax.grad(lambda x:u_dict(x, params)[i],0)(t)  # derivative part of the ODE
                      for i in range(theta.shape[0])]]).T
    f = udt-self.Tmax*(theta[:,[0]] +   # right-hand side of the ODE
            (theta[:, 1:] @ jnp.exp(u_dict(t, params))[:, jnp.newaxis]))
    k = jnp.sum((params['eq_params']["theta"])**2)/jnp.size(params['eq_params']["theta"])  # penalization for sparsity of parameter
    return f + self.J3*k  # Loss associated to the ODE for input t in PINN u with parameter params


class JINN_GLV:
    '''
        Class implementing the Lotka-Volterra PINN using jinns
        Can and will be generalized to every coherent equation
    '''    

    def __init__(self, n, Tmax, layers, batch_size, key, N_0=None, two_step=True, verbose=False, theta=None, sec_layers=None):
        '''
            Initialisation of the PINN, with the data and the sampling objects
            n = int : number of timestamps for ODE penalization
            Tmax = int : time at then end of the experiment (we consider experiment starts at time 0)
            layers = Array[nn.module|fun] : layers of the neural network. Must have 1 or theta.size+1 input and Ns outputs
            batch_size = int : number of timestamps for data penalization
            key = KeyObject : random key
            N_0 = jnp.Array[float](Ns) : initial conditions
            two_step = Bool : whether the algorithm at hand only needs a 'forward' learning (only learning nn parameters, not eq) 
            verbose = Bool : whether to print info
            theta = jnp.array[float](Ns+1 x Ns) : parameter of the equation (initial parameter in the inverse case)
            sec_layers = Array[nn.module|fun] : layers of the second neural network for periodicity. Must have 1  input and Ns outputs
       '''
        self.key, self.subkey = random.split(key)
        self.eqx_list = layers
        self.sec_eqx_list = sec_layers
        self.n = n
        self.batch_size = batch_size
        self.tmin, self.tmax = 0, 1  #as we use time normalization
        self.Tmax = Tmax
        self.train_data = jinns.data.DataGeneratorODE(  # creates batches of timestamps to be used for ODE Loss
            self.subkey, self.n, self.tmin, self.tmax, self.batch_size, method='uniform')
        self.validation_data = jinns.data.DataGeneratorODE(self.key, self.n, self.tmax, 2*self.tmax, self.batch_size, method='uniform')
        self.key, self.subkey = random.split(self.key)
        self.u = jinns.utils.create_PINN(self.subkey, self.eqx_list, "ODE")
        self.init_nn_params = self.u.init_params()
        self.theta = theta  # initial theta, evolving at each iteration in one-step method
        self.verbose = verbose
        self.N_0 = N_0
        self.two_step = two_step
        self.key, self.subkey = random.split(self.key)
        self.params = {}
        self.params["eq_params"] = {"theta": self.theta}
        if self.sec_eqx_list is not None:
            self.sec_u = jinns.utils.create_PINN(self.subkey, self.sec_eqx_list, "ODE")
            self.params["nn_params"] = { "1": self.init_nn_params,
                                         "2": self.sec_u.init_params() }
            self.u_dict = Callable_dict()
            self.u_dict["1"] = self.u
            self.u_dict["2"] = self.sec_u
        else:
            self.sec_u = None
            self.params["nn_params"] = self.init_nn_params
        self.Tj = None
        self.Uij = None
        self._already_running = False


    def __call__(self, t, params=None):
        return self.forward(t, params)

    def forward(self, t, params=None):
        '''
            Given the input and possibly a set of parameters, this function returns the output of the network
        '''
        t = jnp.array(t)
        if params == None:
            params = self.params
        if self.sec_u is None:
            return self.u(t, params["nn_params"])
        else :
            return self.u_dict(t, params)
    
    def set_data(self, data): 
        '''
            function to put data inside the PINN in jinns
        '''
        pass


    def train(self, n_iter:int, lambda1=0.1, lambda2=1, lambda3=5, 
              lr=1e-3, mode="simple", loss_class=GLV_loss, patience=None,
              data=None, sec_opt=None, alter=None):
        '''
            n_iter = int : number of batches to train the network
            lambda_k = float : coefficient in front of the k-loss in the total loss. Influences the importance given to each part for the learning
            lr = float : learning rate for the Adam optimizer
            mode = str : mode for the training. Can be 'simple' or 'validation' as for now
            loss_class = Class : Class for the loss. Must contain an init and an evaluate method
            patience = int : if in mode validation, the number of iterations where the loss is under the threshold before training is stopped
            data = (jnp.array[float](batch_size), jnp.array[float](batch_sizexNs)) : tuple containing the list of sampled time points and the data associated with it 
            sec_opt = optax optimizer : second optimizer for inverse problem parameter optimization in inverse problem mode
            alter = (int, int) : number of iteration for the first and second optimizer in inverse problem mode
        '''
        tx = optax.adam(learning_rate=lr)
        self.params["eq_params"] = {"theta": self.theta}
        observations = None
        if not(data is None or self._already_running):
            self.Tj = data[0]
            self.Uij = data[1]
            self._already_running = True
        if self._already_running or data is not None: # if self.data[1] is None: or (self.data[1]!=data[1]).any():
            observations = jinns.data.DataGeneratorObservations(self.key,  #creates batches of timestamps and data associated for Data Loss
                                                        self.batch_size,
                                                        self.Tj,
                                                        self.Uij)


        if (self.two_step or alter is None) and self.sec_u is None:
            loss_function = loss_class(self.Tmax)  # Loss function
            loss = jinns.loss.LossODE(self.u,
                                {"dyn_loss":lambda1, "initial_condition":lambda2, "observations":lambda3},
                                loss_function, initial_condition=(self.tmin, self.N_0))
        elif not(self.two_step) and self.sec_u is None and alter is not None:  # Inverse with parameter optimization inside jinns.solve
            loss_function = loss_class(self.Tmax)  # Loss function with theta penalty
            loss = jinns.loss.LossODE(self.u,
                                {"dyn_loss":lambda1, "initial_condition":lambda2, "observations":lambda3},
                                loss_function, initial_condition=(self.tmin, self.N_0),
                                                 derivative_keys={"dyn_loss":["eq_params", "nn_params"], 
                                                 "initial_conditions":["nn_params"], 
                                                 "observations":["nn_params"]})
        elif self.two_step:
            loss_function = GLV_loss_duo(self.Tmax)
            loss = jinns.loss.LossODE(self.u_dict,
                                {"dyn_loss":lambda1, "initial_condition":lambda2, "observations":lambda3},
                                loss_function, initial_condition=(self.tmin, self.N_0))
        else:
            raise Exception("TBD : try in the other modes.")

        

        if mode == "validation":
            validation = ValidationLoss(
            loss = loss, # a deep copy of `loss` will be done internally
            validation_data = self.validation_data,
            validation_param_data = None,
            validation_obs_data = observations,
            call_every=50,
            early_stopping=True,
            patience=patience)
        else:
            validation = None

        if alter is None:
            out = solve(n_iter=n_iter,
                            init_params = self.params,
                            data=self.train_data,
                            optimizer=tx,
                            loss=loss,
                            obs_data=observations,
                            validation=validation,
                            print_loss_every=1000,
                            verbose=self.verbose)
        else:    
            optimizers = Optimizers(tx, sec_opt, alter)
            out = solve_alternate(n_iter=n_iter,
                            init_params = self.params,
                            data=self.train_data,
                            optimizers=optimizers,
                            loss=loss,
                            obs_data=observations,
                            validation=validation,
                            print_loss_every=20000)
        self.params, total_loss_list, loss_by_term_dict, loss = out[0:4]

        return total_loss_list, loss_by_term_dict, out[-2]
        
        #  tbd : tolerance mode, adaptative learning mode

    def save_pinn(self, path):
        jinns.utils.save_pinn(path, self.u, self.params,
                      {"eqx_list":self.eqx_list, "type": "ODE"}) 
        return True

    def init_params(self, key):
        Ns = jnp.size(self.N_0)
        self.theta = jax.random.ball(key, 1, 2, (Ns,Ns+1))[:,:,0]
        self.params["nn_params"] = self.u.init_params()
        self.params["eq_params"] = {"theta": self.theta}

    def get_weights(self):
        return self.params["nn_params"]

    def get_theta(self):
        return self.params["eq_params"]["theta"]



class Forward_Problem_GLV:
    # Class implementing the training of the parameters of the neural network such that it approaches a function coherent with data and ODE
    def __init__(self, theta, Ns, Nf, batch_size, Tmax, N_0=None, layers=None):
        self.key = random.PRNGKey(123456789)
        self.theta = theta
        if layers is None or layers==1 :
            self.layers = [[eqx.nn.Linear, 1, 2*Ns],
                    [PSnake, 2.],
                    [eqx.nn.Linear, 2*Ns, 7*Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, 7*Ns, 7*Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, 7*Ns, Ns]]
            depth=2
        elif layers==2:
            self.layers = [[[eqx.nn.Linear, 1, Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, Ns, 7*Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, 7*Ns, 7*Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, 7*Ns, Ns]],

                    [[eqx.nn.Linear, 1, Ns],
                     [eqx.nn.Linear, Ns, 3*Ns],
                     [PSinc, 0.],
                     [eqx.nn.Linear, 3*Ns, Ns]]
                     ]
            depth=3
        else:
            depth = 0
            stack = [(self.layers, 1)]
            while stack:
                curr_list, curr_depth = stack.pop()
                if isinstance(curr_list, list):
                    depth = max(depth, curr_depth)
                    for item in curr_list:
                        stack.append((item, curr_depth + 1))
        
        if depth==2:  # Cas où l'on utilise qu'un réseau
            self.pinn = JINN_GLV(Nf,Tmax, self.layers, batch_size, self.key, theta=self.theta, N_0=N_0)
        else:  #Cas où l'on utilise 2 réseaux
            self.pinn = JINN_GLV(Nf,Tmax, self.layers[0], batch_size, self.key, theta=self.theta, N_0=N_0, sec_layers=self.layers[1])

    def evaluate(self, epochs, lambda1, lambda2, lambda3, mode="simple", data=None, lr=1e-3):
        return self.pinn.train(epochs,lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, 
                               mode=mode, patience=10, data=data, lr=lr)
    
    def change_theta(self, theta=None):
        self.pinn.theta = theta

    def get_params(self):
        return self.pinn.params
    
    def set_params(self, params=None):
        self.pinn.params = params
        if params is not None:
            self.change_theta(params["eq_params"]["theta"])
        else:
            self.change_theta()


class Inverse_Problem_GLV:
    # Class implementing the training of the parameters of the neural network and the ODE such that it approches a function coherent with data
    def __init__(self, theta_ini, Ns, Nf, batch_size, Tmax, N_0=None, layers=None):
        self.key = random.PRNGKey(123456789)
        self.theta = theta_ini
        self.N_0 = N_0
        self.Nf = Nf
        self.Ns = Ns
        self.Tmax = Tmax
        self.layers = layers
        if layers is None:
            self.layers = [[eqx.nn.Linear, 1, 2*Ns],
                    [PSnake, 10.],
                    [eqx.nn.Linear, 2*Ns, 7*Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, 7*Ns, 7*Ns],
                    [jax.nn.tanh],
                    [eqx.nn.Linear, 7*Ns, Ns]]
        self.pinn = JINN_GLV(Nf, Tmax, self.layers, batch_size, self.key, theta=self.theta, N_0=N_0, two_step=False)


    def run(self, epochs_init, n_iter, data, lambda1, lambda2, lambda3, prox_coef, alter):
        lr = float(1 / (2* (self.Tmax ** 2) * (1 + 1/(self.Nf*self.Ns)*jnp.linalg.norm(jnp.exp(data[1])) ** 2)))
        sec_opt = proximal_gradient_optax(learning_rate=lr, lreg=prox_coef, proximal=prox_ridge)
        out = self.pinn.train(epochs_init, 0., lambda2, lambda3, mode="simple", data=data)
        out = self.pinn.train((alter[0]+alter[1])*n_iter, lambda1=lambda1, lambda2=lambda2, lambda3=lambda3, 
                              mode="simple", data=data, sec_opt=sec_opt, alter=alter)
        print(f"Individual losses: { {key: f'{val[-1]:.5f}' for key, val in out[1].items()} }")
        return *out, self.pinn.get_theta(), self.layers



def _cost_function_GLV(theta, data):
    # theta in the form of a (Ns,(Ns+1)) jnp array
    # data in the form of a tuple of 2 jnp (Ns, Nf) arrays and 1 float
    df, expf, tmax, scaling = data
    mu = theta[:,[0]]
    A = theta[:,1:]
    sum_norm = jnp.sum((df-tmax*(mu+A@expf))**2)/(df.shape[0]*df.shape[1])*scaling
    return sum_norm

class GeneralisedSmoothingJINN():

    # Class implementing the Generalised Smoothing spline
    def __init__(self, Ns, Tmax, N_0, Nf, batch_size, key, layers=None, 
                 lambd0 = 0.01, lambd1: float = 10 ** (-2.5), lambd2: float = 10 ** 2.5, 
                 iterationMax: int = 500, errMax: float = 1e-5):
        
        self.lam0= lambd0
        self.lam1 = lambd1
        self.lam2 = lambd2
        self.iterationMax = iterationMax
        self.errMax = errMax
        self.Ns = Ns
        self.Nf = Nf
        self.batch_size = batch_size
        self.N_0 = N_0
        self.Tmax = float(Tmax)
        self.pg = jo.ProximalGradient(fun=_cost_function_GLV, prox=jo.prox.prox_ridge,
                                    stepsize=0., maxiter=15000, tol=50, jit=True, verbose=False)
        
        self.theta = jax.random.ball(key, 1, 2, (self.Ns,self.Ns+1))[:,:,0]
        self.problem = Forward_Problem_GLV(self.theta, self.Ns, self.Nf,
                                           batch_size, self.Tmax,
                                           N_0=jnp.log(self.N_0), layers=layers)
        self.u = self.problem.pinn

    def set_N0(self, N0):
        self.N_0 = N0
        self.problem.pinn.N_0 = jnp.log(N0)

    def set_lam0(self, lam0):
        self.lam0=lam0
    
    def set_lam2(self, lam2):
        self.lam2=lam2

    def init_params(self, key):
        self.problem.pinn.init_params(key)


    def run_alternate(self, epochs_init, epochs, data, prox_coef=1e-1, verbose=False):

        data_ = data.copy()
        times = jnp.linspace(0., 1., self.Nf)
        self.pg.verbose = verbose
        self.pg.stepsize = float(1 / (2* (self.Tmax ** 2) * 
                                (1 + 1/(self.batch_size*self.Ns)*jnp.linalg.norm(jnp.exp(data_[1])) ** 2)))
        data_[0] = (data_[0]/self.Tmax)[:,np.newaxis]
        
        def alternate_loop(carry):
            _data_c, _pinn_c, _metrics_c, _prox_c, _condi_c, i = carry
            errprec, err = _metrics_c
            # self.problem.pinn.params = params
            if type(_pinn_c.lambda0) == tuple:  # three point polynomial curve for GLV-loss coefficient to follow trhough learning
                a = (2*(_pinn_c.lambda0[0]+_pinn_c.lambda0[2])-4*_pinn_c.lambda0[1])/(_condi_c.iterMax**2)
                b = (-(3*_pinn_c.lambda0[0]+_pinn_c.lambda0[2])+4*_pinn_c.lambda0[1])/_condi_c.iterMax
                c = _pinn_c.lambda0[0]
                curr_lam0 = abs(a*i**2 + b*i + c)
            else:
                curr_lam0 = _pinn_c.lambda0

            df = vmap(jax.jacrev(lambda x:self.problem.pinn.forward(x, _pinn_c.params)))(_data_c.time).T
            expf = jnp.exp(vmap(lambda x:self.problem.pinn.forward(x, _pinn_c.params))(_data_c.time)).T
            theta = _pinn_c.params["eq_params"]["theta"]
            res = self.pg.run(theta, hyperparams_prox=_prox_c.prox_coef, data=(df, expf, _data_c.Tmax, 1e0))
            theta = res.params
            _pinn_c.params["eq_params"]["theta"] = theta
            self.problem.set_params(_pinn_c.params)
            loss, loss_dict, _ = self.problem.evaluate(epochs, curr_lam0, _pinn_c.lambda1, _pinn_c.lambda2, 
                                                       data=None, lr=1e-3, mode="validation")
            _pinn_c.params["nn_params"] = self.problem.pinn.get_weights()
            #print("Temps optim u =", chrono() - temps)

            curr_sol = vmap(lambda t:
                            self.problem.pinn.forward(t, _pinn_c.params)[:,jnp.newaxis])(_data_c.data[0]/_data_c.Tmax)[:,:,0]
            errprec = err
            err = _cost_function_GLV(theta, (df, expf, _data_c.Tmax, 1e0))
            jax.lax.cond(i%1==0,
                         lambda _: jax.debug.print("Iteration: {}, Error from GLV only : {}", 
                                                   i, jnp.abs((errprec-err)/err)), 
                         lambda _ :_, None)
            if verbose:
                jax.debug.print("{}", theta[0,0])
                # jax.debug.print("Individual losses: { {key: f'{val[-1]:.5f}' for key, val in loss_dict.items()} }")
            self.problem.set_params()
            return _data_c, _pinn_c, (errprec, err), _prox_c, _condi_c, i+1
        

        def condi(carry):
            _data_c, _pinn_c, _metrics_c, _prox_c, _condi_c, i = carry
            errprec, err = _metrics_c
            return jax.lax.cond(jnp.abs((errprec-err)/errprec)>_condi_c.errMax,
                                lambda _:jax.lax.cond(i<_condi_c.iterMax, lambda _:True, lambda _:False, None), 
                                lambda _:False, None)

        def init_loop():
            self.problem.evaluate(epochs_init, 0., self.lam1, self.lam2, data=data_)
            return self.problem.get_params()

        self.problem.set_params(jax.jit(init_loop)())

        data_c = Data_Container(times, data_, self.Tmax)
        pinn_c = Pinn_Container(self.lam0, self.lam1, self.lam2, self.problem.get_params())
        metrics_c = (20000., 5000.)
        prox_c = Proximal_Container(prox_coef)
        condi_c = Condition_Container(self.errMax, self.iterationMax, epochs)
        

        carry = (data_c, pinn_c, metrics_c, prox_c, condi_c, 0)

        _data_c, _pinn_c, (errprec, err), _prox_c, _condi_c, i = jax.lax.while_loop(condi, alternate_loop, carry)
        self.problem.set_params(_pinn_c.params)
        # params = self.problem.pinn.params
        self.u = self.problem.pinn
        self.u.params = _pinn_c.params
        return _pinn_c.params, err
