import numpy as np
import matplotlib.pyplot as plt
from spline import bsplineM
from scipy.interpolate import BSpline
from scipy.optimize import least_squares, minimize, fmin_cg, leastsq 
from scipy.sparse import csr_matrix
from scipy.integrate import odeint
import time, progressbar
import jax.numpy as jnp
from jax import grad, jit, jacfwd, lax 

##################

def objective_function(coefs, knots, norder, Phi, Phip, Ycell, Tcell, wts, lambda_val, fn, p):
  """
  Calculate the objective function value for the GSA algorithm.

  Args:
    coefs (ndarray): Coefficient matrix.
    knots (ndarray): Knots vector.
    norder (int): B-spline order.
    Phi (ndarray): Basis function matrix.
    Phip (ndarray): Derivative of basis function matrix.
    Ycell (list): List of response values.
    Tcell (list): List of time values.
    wts (list): List of weight matrices.
    lambda_val (float): Lambda value.
    fn (callable): Function to be used.
    p (float): Parameter value.

  Returns:
    ndarray: Objective function value.

  """
  n_exp = len(Ycell)
  n = len(Ycell[0])
  
  nknots = len(knots)
  deltaT = 1.0 / nknots
  quadvals = jnp.column_stack((knots, jnp.ones(nknots) * deltaT))
  nbasis = nknots + norder - 2
  f = jnp.zeros(n_exp * nknots * n)
  E = []; n1 = 0; n2 = 0; offset_d = 0

  for e in range(n_exp):
    n2 += n * nbasis
    C = jnp.reshape(coefs[n1:n2], (nbasis, n), order='F')
    for i in range(n):
      terminal = offset_d + len(Ycell[e][i]) - 1
      debut = n1 + i * nbasis
      fin = n1 + (i + 1) * nbasis - 1
      PhiT = bsplineM(Tcell[e][i], knots, norder, 0)
      E.append(jnp.diag(wts[e][i]) @ (Ycell[e][i] - PhiT @ C[:, i]))
      offset_d = terminal+1

    M = (Phip @ C - fn['fn'](Phi @ C, p)) * jnp.outer(jnp.sqrt(quadvals[:, 1]),jnp.sqrt(lambda_val)) 
    offset = e * n * nknots

    f = f.at[offset:offset + n * nknots].set(jnp.ravel(M))  
 
    n1 = n2

  E = jnp.hstack(E)
  f = jnp.hstack((E,f))

  return f

################## 
def jac_jax(coefs, knots, norder, Phi, Phip, Ycell, Tcell, wts, lambda_val, fn, p):
  ## Jacobian of the objective function computed using JAX
  jacobian = jacfwd(objective_function)(coefs, knots, norder, Phi, Phip, Ycell, Tcell, wts, lambda_val, fn, p)
  return jacobian


##################

def proxP4(p, i, alpha):
  p[0] = max(0, p[0])
  save = p[i+1]
  n = len(p)
  
  # Shrink
  p[1:n] = np.maximum(p[1:n] - alpha, 0) + np.minimum(p[1:n] + alpha, 0)
  
  # Correction for i+1th term
  p[i+1] = min(0, save)
  
  return p

##############

def MakeMatpmultilog(knots, norder, coefs, n_exp, n, deltaT):
  """
  Make the matrix G and B for the proximal operator.

  Args:
    knots (array-like): Knot vector.
    norder (int): Order of the B-spline basis functions.
    coefs (array-like): Coefficient matrix.
    n_exp (int): Number of experiments.
    n (int): Number of variables.
    deltaT (float): Time step.

  Returns:
    tuple: A tuple containing the matrix G, matrix B, and the maximum absolute eigenvalue L.
  """

  B = np.zeros((n+1, n))
  G = np.zeros((n+1, n+1))

  Phip = bsplineM(knots, knots, norder, 1)
  Phi = bsplineM(knots, knots, norder, 0)

  nbasis = len(knots) + norder - 2

  for j in range(n_exp):
    I = slice(j*nbasis*n, (j+1)*nbasis*n)
    C = np.reshape(coefs[I], (nbasis, n), order='F')
    
    Xlp = Phip @ C        
    X = np.exp(Phi @ C)   
  
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

############################

def Alternate_multi(fn, wts, lambd, lambd2, lambd0, iterationMax, errMax, n, Tcell, Ycell, norder, nknots, rg, lsopts_c, use_jax=True, verbose=0):
    ## Alternate minimization algorithm for a multi-experiment case

    deltaT = 1.0 / nknots  
    knots = np.linspace(rg[0], rg[1], nknots)
    quadvals = np.column_stack((knots, np.ones((len(knots), 1)) * deltaT))
    nbasis = nknots + norder - 2
    
    Phi = bsplineM(knots, knots, norder, 0)
    Phip = bsplineM(knots, knots, norder, 1)
    n_exp = len(Ycell)

    U = []
    for j in range(n_exp):
      for i in range(len(Ycell[j])):
        PhiT = bsplineM(Tcell[j][i], knots, norder, 0)
 
        G = PhiT.T @ PhiT + lambd0 * Phip.T @ Phip
        B = PhiT.T @ Ycell[j][i]
        U.append(np.linalg.solve(G, B))

    U = np.column_stack(U)  # Stack the arrays in U as columns

    startpars = np.zeros((n, n + 1))
    newp = startpars
    newcoefs = U.reshape(nbasis * n_exp * len(Ycell[0]), 1, order='F')
    
    err = 1
    iteration = 0
    

    while err > errMax and iteration < iterationMax:
      coefs = newcoefs
      p = newp
      newp = AccProxGradmultilog(p, knots, norder, coefs, n_exp, lambd2, proxP4, deltaT)

      error_spline_args = (knots, norder, Phi, Phip, Ycell, Tcell, wts, lambd, fn, newp)
     
      if use_jax == True:
        result = least_squares(fun=objective_function, x0=np.squeeze(coefs), method='lm', jac=jac_jax, args=error_spline_args, **lsopts_c)
      else:
        result = least_squares(objective_function, np.squeeze(coefs), args=error_spline_args, **lsopts_c)

      newcoefs = result.x

      iteration += 1
      if np.linalg.norm(coefs) == 0 or np.linalg.norm(p) == 0:
        err = np.inf
      else:
        err = np.linalg.norm(coefs - newcoefs) / np.linalg.norm(coefs) + np.linalg.norm(p - newp, 'fro') / np.linalg.norm(p)
   
      if verbose>=1: 
        print((iteration, err))

    return newp, newcoefs

############################

def AccProxGradmultilog(p0, knots, norder, coefs, n_exp, alpha2, proxP4, deltaT):
  """
  Perform accelerated proximal gradient descent for multiclass logistic regression.

  Args:
    p0 (ndarray): Initial parameter vector of shape (n,).
    knots (ndarray): Knots for the spline basis functions.
    norder (int): Order of the spline basis functions.
    coefs (ndarray): Coefficients for the spline basis functions.
    n_exp (int): Number of experiments.
    alpha2 (float): Regularization parameter.
    proxP4 (function): Proximal operator for the P4 penalty.
    deltaT (float): Time step size.

  Returns:
    ndarray: Updated parameter vector of shape (n,).
  """
  n = p0.shape[0]
  
  G, B, L = MakeMatpmultilog(knots, norder, coefs, n_exp, n, deltaT)

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
      z = proxP4(z - dt * grad, i, alpha2 * dt)
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

################

def plot_results(nknots, rg, norder, n, newcoefs, Tcell, Ycell, newp, namespecies, funode):

  n_exp = len(Ycell)
  knots = np.linspace(rg[0], rg[1], nknots)
  nbasis = len(knots) + norder - 2
  ##
  nplot = 1
  offset = 0  # Initialize offset
  
  for j in range(n_exp):
    plt.figure(nplot, figsize=(14,7))
    nplot += 1

    end_vec = offset + nbasis * n
    C = np.reshape(newcoefs[offset:end_vec], (nbasis, n), order='F')
    Phi = bsplineM(knots, knots, norder, 0)

    offset = end_vec
    yest = np.dot(Phi, C)
    yest0 = yest[0, :]

    pathest = odeint(funode, yest0, knots, args=(newp,), rtol=1e-7, atol=1e-9)

    for i in range(len(Ycell[j])):
      plt.subplot(3, 2, i+1)
      plt.plot(knots, np.exp(yest[:, i]), 'r', linewidth=2)
      plt.plot(Tcell[j, i], np.exp(Ycell[j, i]), 'b.')
      plt.plot(knots, np.exp(pathest[:, i]), 'k', linewidth=2)

      if i == 0:
        plt.ylabel(namespecies[i], fontsize=13)
        plt.title('Exp. ' + str(j+1) + ': raw data (b.), profiled solution (r-) and reconstructed solution (k) ', fontsize=13)
      else:
        plt.xlabel('t', fontsize=13)
        plt.ylabel(namespecies[i], fontsize=13)

    plt.tight_layout()

  plt.show()

