import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as si
from scipy.linalg import circulant
import pandas as pd
import jax.numpy as jnp


def GLVlogdfdx(z, p):
    y = z.T
    n = p.shape[0]
    K = y.shape[1]
    r = np.zeros((n, n, K))

    for k in range(K):
        r[:, :, k] = np.dot(p[:, 1:(n+1)], np.diag(np.exp(y[:, k])))

    r = np.transpose(r, (2, 0, 1))
    return r

############################

def GLVlogfun(z, p):
    n = p.shape[0]
    GLV = p[:, 0].reshape(-1, 1, order='F') + jnp.dot(p[:, 1:(n+1)], jnp.exp(z.T))
    r = GLV.T
    return r

############################

def GLVlogfunode(y, t, p):
    i = y.shape[0]
    mu = p[:, 0]
    A = p[:, 1:(i+1)]
    r = mu + np.dot(A, np.exp(y))
    return r

############################

class GLV:

    def __init__(self, log=True, norm_time=True) -> None:
        """Initialise the Generalized Lotka-Voltera model

        Args:
            theta (np.array): Matrix with the parameters
            log (bool, optional): Use the log model. Defaults to True.
            norm_time (bool, optional): Normalize the time. Defaults to True.
        """
        self.use_log = log
        self.norm_time = norm_time
        self.t_end = 1

    def model(self, y, t, theta):
        """ODE governing the model

        Args:
            y (np.ndarray): Population of each species at time t
            t (float): Time
            theta (np.ndarray): Matrix with the parameters
                The parameter theta should be on the form
                [[ mu1, a11, a12, ...],
                 [ mu2, a21, a22, ...],
                   ...,
                 [ mun, an1, an2, ...]]

        Returns:
            np.ndarray: Slope of population of each species at time t
        """
        mu = theta[:, 0]
        A = theta[:, 1:]
        return self.t_end * (mu + A @ y) * y

    def model_log(self, y, t, theta):
        """ODE governing the model

        Args:
            y (np.ndarray): Population of each species at time t
            t (float): Time
            theta (np.ndarray): Matrix with the parameters
                The parameter theta should be on the form
                [[ mu1, a11, a12, ...],
                 [ mu2, a21, a22, ...],
                   ...,
                 [ mun, an1, an2, ...]]

        Returns:
            np.ndarray: Slope of population of each species at time t
        """
        mu = theta[:, 0]
        A = theta[:, 1:]
        return self.t_end * (mu + A @ np.exp(y))

    def compute_solution(self, y0: np.ndarray, theta: np.ndarray, t_end: float=10., nt: int=100):
        """Compute the solution of the model

        Args:
            y0 (np.array): initial population of each species, in *non-logarithmic* scale
            theta (np.ndarray): Matrix with the parameters
                    The parameter theta should be on the form
                        [[ mu1, a11, a12, ...],
                         [ mu2, a21, a22, ...],
                         ...,
                         [ mun, an1, an2, ...]]
            t_end (float, optional): Final time of the simulation. Defaults to 10.
            nt (int, optional): Number of time steps. Defaults to 100.

        Returns:
            tuple: tuple with the times and the solutions at these times
        """
        if self.norm_time:
            self.t_end = t_end
            t = np.linspace(0, 1, nt)
        else:
            self.t_end = 1
            t = np.linspace(0, t_end, nt)
        if self.use_log:
            return self.t_end * t, np.exp(np.array(si.odeint(self.model_log, np.log(y0), t, args=(theta,))))
        else:
            return self.t_end * t, np.array(si.odeint(self.model, y0, t, args=(theta,)))

    def compute_solutions(self, y0_list, theta: np.ndarray, t_end: float=10., nt: float=100):
        """Compute the solution of the model for a list of initial conditions

        Args:
            y0_list (list[np.ndarray]): list of initial populations of each species, in *non-logarithmic* scale
            theta (np.ndarray): Matrix with the parameters
                    The parameter theta should be on the form
                        [[ mu1, a11, a12, ...],
                         [ mu2, a21, a22, ...],
                         ...,
                         [ mun, an1, an2, ...]]
            t_end (float, optional): Final time of the simulation. Defaults to 10..
            nt (float, optional): Number of time steps. Defaults to 100.
        """
        self.t_end = t_end
        t = np.linspace(0, 1, nt)
        if self.use_log:
            return self.t_end * t, np.exp(np.array([si.odeint(self.model_log, np.log(y0), t, args=(theta,)) for y0 in y0_list]))
        else:
            return self.t_end * t, np.array([si.odeint(self.model, y0, t, args=(theta,)) for y0 in y0_list])

    def generate_data(self, y0, theta, number_data=-1, is_random=True, noise=0.0, init_value=False, end_value=False, missing=0, t_end=10., nt=100):
        """Generate data from the simulation

        Args:
            y0 (np.array): initial population, in *non-logarithmic* scale
            theta (np.ndarray): Matrix with the parameters
                    The parameter theta should be on the form
                        [[ mu1, a11, a12, ...],
                         [ mu2, a21, a22, ...],
                         ...,
                         [ mun, an1, an2, ...]]
            number_data (int, optional): Number of data to select. Defaults to -1, if -1, return all data.
            is_random (bool, optional): Select randomly the data. Defaults to True.
            noise (bool, optional): Add noise to generated data. Defaults to False.
            missing (int, optional): Select how many data we want to "miss". Defaults to 0.
            t_end (float, optional): End time of simulation. Defaults to 10.
            nt (int, optional): number of time steps. Defaults to 100.

        Returns:
            tuple: tuple with the times and the solutions at these times
        """
        t, solution = self.compute_solution(y0, theta, t_end, nt)

        # adding noise
        # solution = abs(solution * (1 + np.random.normal(0, noise, solution.shape)))
        s = np.sqrt(np.log(1 + noise**2))*np.ones(solution.shape)
        solution_noisy = np.random.lognormal(np.log(solution) - (1/2)*(s**2),s)
        solution = solution_noisy

        if missing != 0:
            for i in range(theta.shape[0]):
                idx = np.random.choice(solution.shape[0], missing, replace=False)
                solution[idx, i] = np.nan
        if number_data == -1:
            return solution
        else:
            if is_random:
                idx = np.random.choice(solution.shape[0], number_data, replace=False)
                if init_value:
                    if end_value:
                        tdata = np.concatenate((np.array([[0.]]), t[idx].reshape(number_data, 1), np.array([[t_end]])), axis=0)
                        ydata = np.concatenate((y0.reshape(1, len(y0)), solution[idx], solution[-1].reshape(1, len(solution[-1]))), axis=0)
                    else:
                        tdata = np.concatenate((np.array([[0.]]), t[idx].reshape(number_data, 1)), axis=0)
                        ydata = np.concatenate((y0.reshape(1, len(y0)), solution[idx]), axis=0)
                else:
                    if end_value:
                        tdata = np.concatenate((t[idx].reshape(number_data, 1), np.array([[t_end]])), axis=0)
                        ydata = np.concatenate((solution[idx], solution[-1].reshape(1, len(solution[-1]))), axis=0)
                    else:
                        tdata = t[idx].reshape(number_data, 1)
                        ydata = solution[idx]
            else:
                # steps = np.floor(len(t) / number_data+1).astype(int)
                # tdata = np.linspace(0, t_end, number_data+1).reshape(number_data+1, 1)
                # ydata = np.concatenate((solution[0::steps, :], solution[-1].reshape(1, len(solution[-1]))), axis=0)
                nb_points = number_data + (1 if init_value else 0) + (1 if end_value else 0)
                tdata, ydata = self.compute_solution(y0, theta, t_end, nb_points)
                tdata = tdata.reshape(nb_points, 1)

            return tdata, ydata

    def plot_solution(self, t, solution, labels=None, title=None, filename=None):
        """Plot the solution of the model

        Args:
            t (np.ndarray): Time array
            solution (np.ndarray): Solution array
            labels (list, optional): Labels of each species. Defaults to None.
            title (str, optional): Title of the plot. Defaults to None.
            filename (str, optional): Name of the file to save the plot. Defaults to None.
        """

        plt.figure(figsize=(10, 5))
        if labels is None:
            labels = ['Species ' + str(i) for i in range(solution.shape[1])]
        for i in range(solution.shape[1]):
            plt.plot(t, solution[:, i], label=labels[i])
        plt.legend()
        plt.xlabel('Time')
        plt.ylabel('Population')
        if title is not None:
            plt.title(title)
        if filename is not None:
            plt.savefig(filename)

def save_data(filename, times, data):
    """Save the data in a csv file

    Args:
        filename (str): Name of the file to save the data
        times (np.ndarray): Time array
        data (np.ndarray): Data array with the population of each species at each time of the time array
    """
    df = pd.DataFrame()
    if not os.path.exists(filename):
        experiment_number = 1
    else:
        experiment_number = pd.read_csv(filename)['#Exp'].max() + 1
    df['#Exp'] = experiment_number * np.ones_like(times[:,0], dtype=int)
    df['t'] = times[:,0]
    for i in range(data.shape[1]):
        df['N' + str(i)] = data[:, i]
    df.sort_values(by=['#Exp', 't'], inplace=True)
    if not os.path.exists(filename):
        df.to_csv(filename, index=False)
    else:
        df.to_csv(filename, index=False, mode='a', header=False)

def data_to_matrix(experiment_number: int, times: np.ndarray, data: np.ndarray):
    """Save the data in a csv file

    Args:
        experiment_number (int): number of the experiment
        times (np.ndarray): Time array
        data (np.ndarray): Data array with the population of each species at each time of the time array
    """
    df = pd.DataFrame()
    df['#Exp'] = experiment_number * np.ones_like(times[:,0], dtype=int)
    df['t'] = times[:,0]
    for i in range(data.shape[1]):
        df['N' + str(i)] = data[:, i]
    df.sort_values(by=['#Exp', 't'], inplace=True)
    return df.to_numpy()

def generate_parameters(Ns: int):
    """Generate random parameters for the model

    Args:
        Ns (int): Number of species

    Returns:
        np.ndarray: Matrix with the parameters
    """
    # defines random vector of size n that sums to 0
    v = np.random.randn(Ns - 2)

    v = np.concatenate((v, np.array([0 - np.sum(v)])))
    v = np.concatenate((np.array([0.]),v))

    A = 0.05* circulant(v)
    A = A - A.T
    # A += np.random.random(size=A.shape)

    theta = np.concatenate((np.zeros((Ns, 1)), A), axis=1)
    return theta

def generate_random_matrix(size, null_percentage=0.2):
    """Generate a random matrix with a given size and a given percentage of null values

    Args:
        size (int): Size of the matrix
        null_percentage (float, optional): Percentage of null values. Defaults to 0.2.

    Returns:
        np.ndarray: Generated matrix
    """
    # Create a square matrix of zeros
    matrix = np.zeros((size, size))

    # Set non-diagonal elements to non-null values
    non_diagonal_indices = np.where(~np.eye(size, dtype=bool))
    num_non_null = int(size**2 * (1 - null_percentage))
    non_null_indices = np.random.choice(len(non_diagonal_indices[0]), num_non_null, replace=False)
    matrix[non_diagonal_indices[0][non_null_indices], non_diagonal_indices[1][non_null_indices]] = np.random.random(num_non_null)

    matrix += np.random.random(size=matrix.shape)

    return matrix

def generate_parameters_convergent(Ns: int, n_try_max: int=100, null_percentage=0.3):
    """Generate random parameters for the model that lead to a convergent solution

    Args:
        Ns (int): Number of species
        n_try_max (int, optional): Number of try. Defaults to 100.
        null_percentage (float, optional): Percentage of null values in teh matrix A. Defaults to 0.3.

    Returns:
        np.ndarray: Generated parameters
    """

    n_try = 0
    while n_try < n_try_max:
        mu = np.random.rand(Ns)
        A_diag = np.random.uniform(-1, -0.1, Ns)
        A = generate_random_matrix(Ns, null_percentage=null_percentage)
        A = A - A.T
        A[np.where(np.eye(Ns))] = A_diag
        x = np.linalg.solve(A, -mu)
        if np.any(x < 0):
            pass
        else:
            D = np.diag(x)
            candicate_def_neg = A @ D + D @ A.T
            if np.all(np.linalg.eigvals(candicate_def_neg) < 0):
                theta = np.concatenate((mu.reshape(Ns, 1), A), axis=1)
                return theta
        n_try += 1
    raise ValueError("Could not find a convergent solution")