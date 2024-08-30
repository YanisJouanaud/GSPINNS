from scipy.interpolate import BSpline
import numpy as np

def bsplineM(x, breaks, norder=4, nderiv=0):
  """
  Computes values or derivatives of B-spline basis functions
  
  Arguments:
  x         ... Argument values for which function values are computed
  breaks    ... Increasing knot sequence spanning argument range
  norder    ... Order of B-spline (one greater than degree) max = 19
                Default 4.
  nderiv    ... Order of derivative required, default 0.
  sparsewrd ... if 1, return in sparse form
  
  Returns:
  bsplinemat ... length(X) times number of basis functions matrix
                 of Bspline values
  """
  # Check dimensions of x and set up as a row vector
  if x.ndim > 1:
    raise ValueError("Argument X is not a vector.")
  x = x.flatten()

  #n = len(x)
  
  nbreaks = len(breaks)
  if nbreaks < 2:
    raise ValueError("Number of knots less than 2.")
  if any(np.diff(breaks) < 0):
    raise ValueError("BREAKS are not strictly increasing.")

  # Check norder
  if norder < 1:
    raise ValueError("Order of basis less than one.")

  # Check NDERIV
  if nderiv < 0:
    raise ValueError("NDERIV is negative")
  if nderiv >= norder:
    raise ValueError("NDERIV cannot be as large as order of B-spline.")

  # Check X
  if (np.diff(x)).size > 0 and np.min(np.diff(x)) < 0:
    x = np.sort(x)
    sortwrd = True
  else:
    sortwrd = False
  if x[0] - breaks[0] < -1e-10 or x[-1] - breaks[-1] > 1e-10:
    raise ValueError("Argument values out of range.")

  # Construct B-spline basis functions using scipy.interpolate.BSpline
  knots = np.concatenate(([breaks[0]] * (norder-1), breaks, [breaks[-1]] * (norder-1)))
  spline = BSpline(knots, np.eye(len(knots) - norder), norder-1)

  # evaluate the B-spline basis functions
  bsplinemat = spline(x, nu=nderiv)

  if sortwrd:
    bsplinemat = bsplinemat[np.argsort(x)]
  
  return bsplinemat


