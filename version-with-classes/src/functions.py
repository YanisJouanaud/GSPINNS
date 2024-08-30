import numpy as np
import matplotlib.pyplot as plt
from spline import bsplineM
from scipy.optimize import least_squares
from scipy.integrate import odeint
import ExperimentData as data
import jax.numpy as jnp
from jax import grad, jit, jacfwd
import sys
sys.path.append("src")
from pinn_glv import PhysicsInformedNN

def proxP4(p, i, alpha):
    p[0] = max(0, p[0])
    save = p[i+1]
    n = len(p)
    
    # Shrink
    p[1:n] = np.maximum(p[1:n] - alpha, 0) + np.minimum(p[1:n] + alpha, 0)
    
    # Correction for i+1th term
    p[i+1] = min(0, save)
    
    return p

############################
class GeneralisedSmoothingSpline:
    
    def __init__(self, lambd0: float = 0.01, lambd1: float = 10 ** (-2.5), lambd2LS: float = 10 ** 2.5, iterationMax: int = 500, errMax: float = 1e-5, norder: int = 3, fold: int = 2):
        self.lam0 = lambd0
        self.lam1 = lambd1
        self.lam2LS = lambd2LS
        self.iterationMax = iterationMax
        self.errMax = errMax
        self.norder = norder
        self.fold = fold
        

    def load_data(self, filename: str, scale_factor: float = 10**0):
        self.experiment_data = data.ExperimentData(filename)
        self.nb_species = self.experiment_data.N_species
        self.namespecies = self.experiment_data.get_species_name()
        self.scale = scale_factor * np.ones(len(self.namespecies))
        self.n_exp = self.experiment_data.n_exp
        #
        self.Tcell, self.path_cell, self.Ycell = self.experiment_data.create_cell_arrays_for_data_storing(self.scale)
        temps = np.sort(np.unique(self.experiment_data.get_time()))
        tini = temps[0]
        horizT = temps[-1] - tini
        tspan = np.arange(tini, tini + horizT + 1)
        self.rg = [np.min(temps), np.max(temps)]
        self.nknots = self.fold * (len(tspan) - 1) + 1
        self.nbasis = self.nknots + self.norder - 2


    ############################
        
    def run_alternate_multi(self, fn, reduce_tol: bool=False, verbose: int=0, lsopts_c:dict={'verbose': 0}, jac_analytical: bool = False):
        """Run the alternate minimization algorithm
        Args:
        ----
            fn (dict): Model function that corresponds to the ODE system with key 'fn'
            reduce_tol (bool): If True, the algorithm will reduce the tolerance adaptively
            verbose (int): If >0, the algorithm will print some information
            lsopts_c (dict): optimization parameters for least squares solver
            jac_analytical (bool): If True, the algorithm will use the analytical jacobian specified in fn['dfdx']

        Returns:
        -------
            tuple:
                - newp : optimized parameters corresponding to the model
                - newcoefs : optimized coefficients for the spline
        """
        norder = self.norder
        Ycell = self.Ycell
        Tcell = self.Tcell
        n_exp = self.n_exp
        nbasis = self.nbasis
        ##
        self.wts = np.ones_like(Ycell)
        deltaT = 1.0 / self.nknots  # =T/nknots*1/T pour prendre la valeur moyenne de l'integrale
        knots = np.linspace(self.rg[0], self.rg[1], self.nknots)


        ########################################
        # Step 0
        ########################################
        Phi = bsplineM(knots, knots, norder, 0)
        Phip = bsplineM(knots, knots, norder, 1)

        U = []
        for j in range(n_exp):
            for i in range(len(Ycell[j])):
                PhiT = bsplineM(Tcell[j][i], knots, norder, 0)

                G = PhiT.T @ PhiT + self.lam0 * Phip.T @ Phip
                B = PhiT.T @ Ycell[j][i]
                U.append(np.linalg.solve(G, B))

        U = np.column_stack(U)  # Stack the arrays in U as columns

        startpars = np.zeros((self.nb_species, self.nb_species + 1))
        newp = startpars
        newcoefs = U.reshape(nbasis * self.n_exp * len(Ycell[0]), 1, order='F')

        err = 1
        iteration = 0

        err_evolution = []
        iterations_err_max_change = []

        while err > self.errMax and iteration < self.iterationMax:

            ########################################
            # Step 1 :
            ########################################

            coefs = newcoefs
            p = newp
            newp = self.AccProxGradmultilog(p, knots, coefs, deltaT)

            ########################################
            # Step 2 :
            ########################################

            error_spline_args = (knots, Phi, Phip, fn, newp)
            x0 = np.squeeze(coefs)
            if jac_analytical == False:
                result = least_squares(fun=self.objective_function, x0=x0, method='lm', jac=self.jac_jax, args=error_spline_args, **lsopts_c)
            else:
                ## Use analytical jacobian (STILL NOT WORKING PROPERLY)
                result = least_squares(fun=self.objective_function, x0=x0, method='lm', jac=self.jacobian, args=error_spline_args, **lsopts_c)

            if verbose>1:
              print((result.nfev,result.njev))

            newcoefs = result.x

            iteration += 1
            if np.linalg.norm(coefs) == 0 or np.linalg.norm(p) == 0:
                err = np.inf
            else:
                err = np.linalg.norm(coefs - newcoefs) / np.linalg.norm(coefs) + np.linalg.norm(p - newp, 'fro') / np.linalg.norm(p, 'fro')

            if verbose: print((iteration, err))
            err_evolution.append(err)
            
            if reduce_tol:
                if iteration % 20 == 0 and iteration > 30:
                    # Check if we have found something better in the 20 last iterations
                    if min(err_evolution[-20:]) != min(err_evolution):
                        self.errMax *= 10
                        iterations_err_max_change.append(iteration)
                        print("Increasing errMax to ", self.errMax)

        
        return newp, newcoefs


    def AccProxGradmultilog(self, p0, knots, coefs, deltaT):
        n = p0.shape[0]
        
        G, B, L = self.MakeMatpmultilog(knots, coefs, n, deltaT)

        p = np.zeros_like(p0)
        for i in range(n):
            pi = p0[i]
            z = p0[i]
            theta = 1
            it = 1
            err = 1
            
            while err > 1e-5 and it < 5000:
                y = (1 - theta) * pi + theta * z
                dt = 1 / (L * theta)
                grad = G @ y - B[:, i]
                z = proxP4(z - dt * grad, i, self.lam1 * dt)
                pinew = (1 - theta) * pi + theta * z
                norm_pi = np.linalg.norm(pi)
                if norm_pi != 0:
                    err = np.linalg.norm(pinew - pi) / norm_pi 
                else:
                    err = np.inf 

                theta = theta * (np.sqrt(theta ** 2 + 4) - theta) / 2
                pi = pinew
                it += 1
            p[i] = pi
        
        return p
    
    def MakeMatpmultilog(self, knots, coefs, n, deltaT):
        B = np.zeros((n+1, n))
        G = np.zeros((n+1, n+1))

        Phip = bsplineM(knots, knots, self.norder, 1)
        Phi = bsplineM(knots, knots, self.norder, 0)

        for j in range(self.n_exp):
            I = slice(j*self.nbasis*n, (j+1)*self.nbasis*n)
            C = np.reshape(coefs[I], (self.nbasis, n), order='F')

            Xlp = Phip @ C
            X = np.exp(Phi @ C)

            K = X.shape[0]
            M1 = K
            #
            M3 = np.sum(Xlp, axis=0)
            #
            M2 = np.sum(X, axis=0)
            #
            N1 = M2.reshape(-1, 1, order='F')
            N2 = X.T @ X
            N3 = X.T @ Xlp

            G += np.vstack((np.hstack((M1, M2)), np.hstack((N1, N2))))
            B += np.vstack((M3, N3))

        G *= 2 * deltaT
        B *= 2 * deltaT

        L = np.max(np.abs(np.linalg.eigvals(G)))

        return G, B, L

    ############################

    def objective_function(self, coefs, knots, Phi, Phip, fn, p):
        Ycell = self.Ycell
        Tcell = self.Tcell
        #
        n = len(Ycell[0])
        
        deltaT = 1.0 / self.nknots
        quadvals = np.column_stack((knots, np.ones(self.nknots) * deltaT))
        
        f = jnp.zeros(self.n_exp * self.nknots * n)
        E = []; n1 = 0; n2 = 0; offset_d = 0

        for e in range(self.n_exp):
            n2 += n * self.nbasis
            C = jnp.reshape(coefs[n1:n2], (self.nbasis, n), order='F')
            for i in range(n):
                terminal = offset_d + len(Ycell[e][i]) - 1
                debut = n1 + i * self.nbasis
                fin = n1 + (i + 1) * self.nbasis - 1
                PhiT = bsplineM(Tcell[e][i], knots, self.norder, 0)
                E.append(self.wts[e][i] * (Ycell[e][i] - PhiT @ C[:, i]))
                offset_d = terminal+1

            M = (Phip @ C - fn['fn'](Phi @ C, p)) * jnp.outer(jnp.sqrt(quadvals[:, 1]),jnp.sqrt(self.lam2LS)) 
            offset = e * n * self.nknots
            #f[offset:offset + n * nknots] = M.flatten()
            f = f.at[offset:offset + n * self.nknots].set(jnp.ravel(M)) 
            
            n1 = n2

        E = jnp.hstack(E)
        ##
        f = jnp.hstack((E,f))
        
        return f

    def jac_jax(self, coefs, knots, Phi, Phip, fn, p):
        return jacfwd(self.objective_function)(coefs, knots, Phi, Phip, fn, p)
  
    def jacobian(self, coefs, knots, Phi, Phip, fn, p):
        Ycell = self.Ycell
        Tcell = self.Tcell
        nbasis = self.nbasis

        n = len(Ycell[0])        
        deltaT = 1.0 / self.nknots
        quadvals = np.column_stack((knots, np.ones(self.nknots) * deltaT))
        
        J = np.zeros((self.n_exp * self.nknots * n, self.n_exp * nbasis * n))
        U = np.zeros((np.sum([len(Ycell[e, i]) for e in range(self.n_exp) for i in range(n)]), nbasis * n * self.n_exp))
        n1 = 0
        n2 = 0
        offset_d = 0

        for e in range(self.n_exp):
            n2 += n * nbasis
            C = np.reshape(coefs[n1:n2], (nbasis, n), order='F')
            for i in range(n):
                terminal = offset_d + len(Ycell[e][i]) - 1
                debut = n1 + i * nbasis
                fin = n1 + (i + 1) * nbasis - 1
                PhiT = bsplineM(Tcell[e][i], knots, self.norder, 0)
                U[offset_d:terminal + 1, debut:fin + 1] = -self.wts[e][i] *  PhiT
                offset_d = terminal+1

            offset = e * n * self.nknots

            z = fn['dfdx'](Phi @ C, p)

            for i in range(n):
                offset_m = offset + i * self.nknots
                for j in range(n):
                    debut = n1 + j * nbasis
                    fin = n1 + (j + 1) * nbasis - 1
                    J[offset_m:offset_m + self.nknots, debut:fin+1] = -np.diag(z[:, i, j] * np.sqrt(quadvals[:, 1])) @ (Phi * np.sqrt(self.lam2LS))
                
                debut = n1 + i * nbasis
                fin = n1 + (i + 1) * nbasis - 1
                J[offset_m:offset_m + self.nknots, debut:fin+1] += np.sqrt(self.lam2LS) * np.diag(np.sqrt(quadvals[:, 1])) @ Phip
        
            n1 = n2        
        ##
        J = np.vstack((U,J))

        return J
    
    ############################

    def plot_results(self, newp, newcoefs, funode, Yexa=None):
        
        n = len(self.Ycell[0])
        knots = np.linspace(self.rg[0], self.rg[1], self.nknots)
        ##
        nplot = 1
        offset = 0  # Initialize offset
        
        for j in range(self.n_exp):
            plt.figure(nplot, figsize=(14,7))
            nplot += 1

            end_vec = offset + self.nbasis * n
            C = np.reshape(newcoefs[offset:end_vec], (self.nbasis, n), order='F')
            Phi = bsplineM(knots, knots, self.norder, 0)
            
            offset = end_vec
            yest = np.dot(Phi, C)
            yest0 = yest[0, :]

            pathest = odeint(funode, yest0, knots, args=(newp,), rtol=1e-7, atol=1e-9)

            for i in range(len(self.Ycell[j])):
                plt.subplot(3, 2, i+1)
                plt.plot(knots, np.exp(yest[:, i]), 'r', linewidth=2, label='profiled solution')
                plt.plot(self.Tcell[j, i], np.exp(self.Ycell[j, i]), 'b.', label='raw data')
                plt.plot(knots, np.exp(pathest[:, i]), 'k', linewidth=2, label='reconstructed solution')
                if Yexa is not None:
                    plt.plot(Yexa[0], Yexa[1][:, i], 'g', linewidth=2, label='exact solution')
                # plt.legend(loc='best', fontsize=13)
                if i == 0:
                    plt.ylabel(self.namespecies[i], fontsize=13)
                    plt.title('Exp. ' + str(j+1) + ': raw data (b.), profiled solution (r-) and reconstructed solution (k) ', fontsize=13)
                else:
                    plt.xlabel('t', fontsize=13)
                    plt.ylabel(self.namespecies[i], fontsize=13)

            plt.tight_layout()
        plt.show()


############################

class GeneralisedSmoothingPINN(GeneralisedSmoothingSpline):

    def __init__(self, lambd0: float = 0.01, lambd1: float = 10 ** (-2.5), lambd2: float = 10 ** 2.5, iterationMax: int = 500, errMax: float = 1e-5):
        self.lam0 = lambd0
        self.lam1 = lambd1
        self.lam2 = lambd2
        self.iterationMax = iterationMax
        self.errMax = errMax

    def run_alternate_multi(self, layers, nb_epochs_init: int, nb_epochs: int, epochs_managment: str="double", func_adapt:str='two', seed: int=1, reduce_tol: bool=False, verbose: int=0):
        """Run the alternate minimization algorithm with PINN
        Args:
        ----
            layers (list): List of integers representing the number of neurons in each layer
            nb_epochs_init (int): Number of epochs for the initial training of the PINN
            nb_epochs (int): Number of epochs for the training of the PINN
            epochs_managment (str): Type of epochs management
            func_adapt (str): Function to adapt the number of epochs
            seed (int): Seed for the random number generator
            reduce_tol (bool): If True, the algorithm will reduce the tolerance adaptively
            verbose (int): If >0, the algorithm will print some information

        Returns:
        -------
            tuple:
                - newp : optimized parameters corresponding to the model
                - newtrajS : optimized coefficients for the spline
                - pinnS : dictionary of trained PINNs
        """

        n_exp = self.n_exp
        Ycell = self.Ycell
        Tcell = self.Tcell

        # definition du pas de temps
        deltaT = 1.0 / self.nknots  # =T/nknots*1/T pour prendre la valeur moyenne de l'integrale
        # definition des différentes abscisses des knots (vector)
        knots = np.linspace(self.rg[0], self.rg[1], self.nknots)

        
        trajS = np.zeros((n_exp,self.nknots, self.nb_species))
        trajlpS = np.zeros((n_exp,self.nknots, self.nb_species)) # getting

        ########################################
        # Step 0
        ########################################

        Phi = bsplineM(knots, knots, self.norder, 0)
        Phip = bsplineM(knots, knots, self.norder, 1)

        for j in range(n_exp):
            U = []
            for i in range(len(Ycell[j])):
                PhiT = bsplineM(Tcell[j][i], knots, self.norder, 0)

                G = PhiT.T @ PhiT + self.lam0 * Phip.T @ Phip
                B = PhiT.T @ Ycell[j][i]
                U.append(np.linalg.solve(G, B))

            U = np.column_stack(U)  # Stack the arrays in U as columns
            trajS[j] = np.exp(Phi @ U)
            trajlpS[j] = Phip @ U

        startpars = np.zeros((self.nb_species, self.nb_species + 1))
        newp = startpars

        err = 1
        iteration = 0

        pinnS = {}

        for j in range(n_exp):
            Tj = Tcell[j][0]
            Tj = Tj.reshape(len(Tj), 1)
            Uji = np.stack(Ycell[j], axis=0).T
            t_f = knots
            t_f = t_f.reshape(len(t_f), 1)
            pinnS[j] = PhysicsInformedNN(t_f, Tj, Uji, layers, renorm=self.rg[1], set_seed=seed, verbose=verbose)

        err_evolution = []
        iterations_err_max_change = []

        while err > self.errMax and iteration < self.iterationMax:

            ########################################
            # Step 1 :
            ########################################

            p = newp
            newp = self.AccProxGradmultilog(p, trajS, trajlpS, proxP4, deltaT)

            ########################################
            # Step 2 :
            ########################################
            newtrajS = np.zeros((n_exp, self.nknots, self.nb_species))
            newtrajlpS = np.zeros((n_exp, self.nknots, self.nb_species))

            for j in range(n_exp):
                # if first iteration, define and train the PINN without given trained weights
                if iteration == 0:
                    pinnS[j].train(nb_epochs=nb_epochs_init, theta=newp, lambda_2=self.lam2, verbose=verbose) # PINN training

                # if not first iteration, define and train PINN with given trained_weigths
                else:
                    if epochs_managment=="basic":
                        nb_epochs_pinn = 100
                        loss_toler = 1e-15
                    elif epochs_managment=="loss_tolerance":
                        nb_epochs_pinn = nb_epochs
                        loss_toler = 1e-3
                    elif epochs_managment=="adaptative_epochs":
                        if func_adapt=='one':
                            nb_epochs_pinn = np.min((1 + np.floor(1e3 * np.exp(.5 * np.log(err_param))),nb_epochs)).astype(int)
                        elif func_adapt=='two':
                            nb_epochs_pinn = np.min((1 + np.floor(1e3 * np.exp(.8 * np.log(err_param))),nb_epochs)).astype(int)
                        else:
                            raise TypeError("func_adapt is not well defined.") 
                        loss_toler = 1e-15
                    elif epochs_managment=="double":
                        if func_adapt=='one':
                            nb_epochs_pinn = np.min((1 + np.floor(1e3 * np.exp(.5 * np.log(err_param))),nb_epochs)).astype(int)
                        elif func_adapt=='two':
                            nb_epochs_pinn = np.min((1 + np.floor(1e3 * np.exp(.8 * np.log(err_param))),nb_epochs)).astype(int)
                        else:
                            raise TypeError("func_adapt is not well defined.")                     
                        loss_toler = 1e-3
                    else:
                        raise TypeError("epochs_managment is not well defined.") 

                    pinnS[j].train(nb_epochs_pinn, theta=newp, lambda_2=self.lam2, verbose=verbose, loss_toler=loss_toler) # PINN training

            # modifying the trajectories with the new trajectories
            newtrajS[j] = pinnS[j].predict(knots.reshape(self.nknots,1))
            newtrajlpS[j] = pinnS[j].predict_log_dt(knots.reshape(self.nknots,1))

            # error computing to change to be consistent with trajectories
            iteration += 1
            if np.linalg.norm(trajS) == 0 or np.linalg.norm(p, 'fro') == 0:
                err = np.inf
                err_param = 1e10
            else:
                # computing the relative error/residual of the iteration
                err = np.linalg.norm(trajS - newtrajS) / np.linalg.norm(trajS) + np.linalg.norm(p - newp, 'fro') / np.linalg.norm(p, 'fro')
                err_param = np.linalg.norm(p - newp, 'fro') / np.linalg.norm(p, 'fro')

            # updating the trajectory (normal)
            trajS = newtrajS
            # updating the time derivative of the trajectory (log)
            trajlpS = newtrajlpS
            if verbose: print((iteration, err))
            err_evolution.append(err)


            if iteration % 20 == 0 and iteration > 30 and reduce_tol:
                    # Check if we have found something better in the 20 last iterations
                    if min(err_evolution[-20:]) != min(err_evolution):
                        self.errMax *= 10
                        iterations_err_max_change.append(iteration)
                        print("Increasing errMax to ", self.errMax)

        return newp, newtrajS, pinnS


    def AccProxGradmultilog(self, p0, trajS, trajlpS, proxP4, deltaT):
        
        G, B, L = self.MakeMatpmultilog(trajS, trajlpS, deltaT)

        p = np.zeros_like(p0)
        for i in range(self.nb_species):
            pi = p0[i]
            z = p0[i]
            theta = 1
            it = 1
            err = 1
        
            while err > 1e-5 and it < 5000:
                y = (1 - theta) * pi + theta * z
                dt = 1 / (L * theta)
                grad = G @ y - B[:, i]
                z = proxP4(z - dt * grad, i, self.lam1 * dt)
                pinew = (1 - theta) * pi + theta * z
                norm_pi = np.linalg.norm(pi)
                if norm_pi != 0:
                    err = np.linalg.norm(pinew - pi) / norm_pi
                else:
                    err = np.inf

                theta = theta * (np.sqrt(theta ** 2 + 4) - theta) / 2
                pi = pinew
                it += 1
            p[i] = pi

        return p
    

    def MakeMatpmultilog(self, trajS, trajlpS, deltaT):
        n = self.nb_species
        n_exp = self.n_exp
        #
        B = np.zeros((n+1, n))
        G = np.zeros((n+1, n+1))

        for j in range(n_exp):    
            Xlp = trajlpS[j] # u_t(knots) (en log)  # should be of size (nknots, n_species)
            X = trajS[j]     # u(knots) (en normal) # should be of size (nknots, n_species)
            
            K = X.shape[0]
            M1 = K
            #
            M3 = np.sum(Xlp, axis=0)
            #
            M2 = np.sum(X, axis=0)
            #
            N1 = M2.reshape(-1,1, order='F')
            N2 = X.T @ X
            N3 = X.T @ Xlp

            G += np.vstack((np.hstack((M1, M2)), np.hstack((N1, N2))))
            B += np.vstack((M3, N3))   
    
        G *= 2 * deltaT
        B *= 2 * deltaT
    
        L = np.max(np.abs(np.linalg.eigvals(G)))
    
        return G, B, L