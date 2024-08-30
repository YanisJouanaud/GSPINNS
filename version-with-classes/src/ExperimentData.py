import numpy as np
import matplotlib.pyplot as plt


class ExperimentData:

    def __init__(self, path, indcol=None) -> None:
        """Load data from csv / txt file

        Args:
            path (str): Path to the data file
            indcol (_, optional): Range of index you want to work with. Defaults to None. If None, all are used
        """
        self.raw_data = np.genfromtxt(path, delimiter=',')
        
        if indcol is None:
            indcol = range(2, self.raw_data.shape[1])
        self.speciesexp = self.raw_data[:, indcol]
        self.time = self.raw_data[:, 1]
        self.N_species = self.speciesexp.shape[1]

        self.species_names = [f'N{i+1}' for i in range(self.N_species)]

        self.ind_exp = np.unique(self.raw_data[:, 0])
        self.n_exp = len(self.ind_exp)

    def get_species(self, species_index=None):
        """Get the data for a specific species

        Args:
            species_index (list, optional): List of indices of the species wanted. Defaults to None. If None, all are used

        Returns:
            np.ndarray: Array of the data for the species
        """
        if species_index is None:
            return self.speciesexp
        else:
            return self.speciesexp[:, species_index]
    
    def get_species_name(self, species_index=None):
        """Get the name of a specific species

        Args:
            species_index (list, optional): List of indices of the species wanted. Defaults to None. If None, all are used

        Returns:
            list: List of the names of the species
        """
        if species_index is None:
            return self.species_names
        else:
            return self.species_names[species_index]
    
    def get_time(self):
        """Get the time data

        Returns:
            np.ndarray: Array of the time data
        """
        return self.time
    
    def experiment(self, index):
        """Get the data for a specific experiment

        Args:
            index (int): Index of the experiment

        Returns:
            np.ndarray: Array of the time data for the experiment
            np.ndarray: Array of the data for the experiment
        """
        ind = np.where(self.raw_data[:, 0] == self.ind_exp[index])[0]
        tt = self.time[ind]
        species = self.speciesexp[ind, :]
        return tt, species
    
    def get_initial_condition(self, exp_index: int):
        """Returns the initial conditions of the given experiment, if it is included in data

        Args:
            exp_index (int): Index of the experiment
        """
        ind = np.where(self.raw_data[:, 0] == self.ind_exp[exp_index])[0]
        return self.speciesexp[ind[0], :]
    
    def create_cell_arrays_for_data_storing(self, scale: np.ndarray):
        """Create cell arrays for data storing

        Args:
            scale (np.ndarray): Array of scaling for each species

        Returns:
            np.ndarray: Array of shape (n_exp, N_species, n_time) with the time data for the experiment
            np.ndarray: Array of shape (n_exp, N_species, n_time) with the data for the experiment
            np.ndarray: Array of shape (n_exp, N_species, n_time) with the log data for the experiment
        """
        Tcell = np.empty((self.n_exp, self.N_species), dtype=object)
        path_cell = np.empty((self.n_exp, self.N_species), dtype=object)
        Ycell = np.empty((self.n_exp, self.N_species), dtype=object)

        for j in range(self.n_exp):
            tt, species = self.experiment(j)
            # t_init = tt[0]

            for i in range(self.N_species):
                obs_pts = np.where(species[:, i] >= 0)[0]
                Tcell[j][i] = tt[obs_pts]
                path_cell[j][i] = species[obs_pts, i]
                Ycell[j][i] = np.log(np.maximum(path_cell[j][i] / scale[i], 1e-8))
        
        Tcell = np.array(Tcell)
        path_cell = np.array(path_cell)
        Ycell = np.array(Ycell)
 
        return Tcell, path_cell, Ycell


    def plot(self, show=False):
        """Plot the data with matplotlib

        Args:
            show (bool, optional): Whether to show the plot. Defaults to False.
        """
        for i in range(self.N_species):
            plt.scatter(self.time, self.speciesexp[:, i], label=self.species_names[i])
        plt.legend(self.species_names)
        plt.xlabel('Time')
        plt.ylabel('Population')
        plt.legend()
        if show:
            plt.show()

