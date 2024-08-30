"""
This code defines two classes for the implementation of a Physics Informed Neural Network (PINN) for the Generalized Lotka-Volterra (GLV) model.
The first one defines the Deep Neural Network (DNN), the second one the PINN procedure, that calls the DNN class
"""
import time
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.autograd import Variable
from collections import OrderedDict

""" === Defining the deep neural network class === """

class DNN(torch.nn.Module):
    def __init__(self, layers, weight_init='xavier'):
        super(DNN, self).__init__()

        # Extracting the number of layers in the network
        self.depth = len(layers) - 1

        # Setting up the activation function for all layers (hyperbolic tangent activation)
        self.activation = torch.nn.Tanh

        # Creating a list to hold layer definitions (linear layers and activation functions)
        layer_list = list()
        for i in range(self.depth - 1):
            # Adding linear layer to the list with appropriate input and output size
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1]))
            )
            # Adding the activation function to the list
            layer_list.append(('activation_%d' % i, self.activation()))

        # Adding the final linear layer to the list with input and output size of the last two layers
        layer_list.append(
            ('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1]))
        )

        # Creating an ordered dictionary to hold the layers and activations in sequence
        layerDict = OrderedDict(layer_list)

        # Creating the neural network model using the ordered dictionary of layers and activations
        self.layers = torch.nn.Sequential(layerDict)

        # Weight initialization
        if weight_init == 'default':
            pass  # Use default weight initialization
        elif weight_init == 'xavier':
            self._initialize_weights_xavier()
        else:
            raise ValueError("Invalid weight_init value. Use 'default' or 'xavier'.")

    def forward(self, x):
        # Forward pass through the neural network
        out = self.layers(x)
        return out

    def _initialize_weights_xavier(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_weights(self):
        # Get the current weights of the neural network
        return self.state_dict()


""" === Setting up the Physics Informed Neural Network === """

class PhysicsInformedNN():


    ##########
    # intialisation of the PINN

    def __init__(self, t_f, Tj, Uji, layers, initial_weights=None, renorm=1.0, set_seed=1, verbose=True):
        """Initialisation of the PINN

        Args:
            t_f (np.ndarray): time array for collocation points (fine grid, matrix of size (N_f,1) )
            Tj (np.ndarray): time array for training data (matrix of size (nb_obs+1 , 1) )
            Uji (np.ndarray): training data  (matrix of size (nb_obs+1 , nb_species) )
            p (np.ndarray): parameter of the GLV model
            layers (list): list of layers for the DNN
            initial_weights (OrderedDict, optional): initial weights for the DNN. Defaults to None.
            renorm (float, optional): renormalisation factor for the time, e.g. 10 if time of simulation goes to 10. Defaults to 1.0, i.e. no renormalisation.
        """
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        # elif torch.backends.mps.is_available():
        #     self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')

        if verbose:
            print('device :', self.device)

        # dealing with inputs : t_f (collocation points t^i_f, i = 1...N_f)
        # and Tj, Uji (training data t_j, u_ji, j = 1...nb_obs+1, i=1...nb_species)
        #
        # t_f should be delt as a torch.tensor variable of size (N_f,1)
        self.t_f = torch.tensor(t_f*(1/renorm), requires_grad=True).float().to(self.device)
        self.final_time = t_f[-1][0]
        # Tj should be delt as a torch.Tensor variable of size (nb_obs+1,1)
        self.Tj = torch.tensor(Tj*(1/renorm), requires_grad=True).float().to(self.device)
        # Uji should be delt as a torch.Tensor varialbe of size (nb_obs+1,nb_species)
        self.Uji = torch.tensor(Uji, requires_grad=True).float().to(self.device)
        self.nb_species = self.Uji.shape[1]

        self.renorm = torch.tensor(renorm, requires_grad=False).float().to(self.device)

        # setting up the Deep Neural Nework with the layers given
        self.layers = layers
        torch.manual_seed(set_seed)
        self.dnn = DNN(self.layers).to(self.device)

        # Setting the initial weights if provided
        if initial_weights:
            self.dnn.load_state_dict(initial_weights)

        # setting up the cost function : MSE
        self.mse_cost_function = nn.MSELoss()

        # setting up the optimizer
        self.optimizer_Adam = torch.optim.Adam(self.dnn.parameters())


    ##########
    # defining the time derivative

    def dt(self, u, t):

        nb_species = u.shape[1] # getting the number of species from the size of u
        u_t = [] # initializing the time derivatives

        # for each species (i), we compute the time derivative ui_t
        for i in range(nb_species):
            ui_t = torch.autograd.grad(u[:, i], t,
                                       grad_outputs=torch.ones_like(u[:, i]),
                                       retain_graph=True,
                                       create_graph=True,
                                       allow_unused=True)[0]
            u_t.append(ui_t) # time derivatives are stored in u_t
        u_t = torch.cat(tuple(u_t), dim=1) # we concateneate all the derivatives
        return u_t

    ##########
    # defining the functional of the ode to minimize later

    def functional_f(self, t, theta):
        # dealing with parameters recovered from matrix p
        #
        # A should be delt as a torch.tensor variable of size (nb_species,nb_species)
        A = torch.as_tensor(theta[:, 1:], dtype=torch.float, device=self.device)
        # mu should be delt as a torch.tensor variable of size (nb_species)
        mu = torch.as_tensor(theta[:, 0], dtype=torch.float, device=self.device)

        u = self.dnn(t)
        u_t = self.dt(u, t)
        ode = u_t - self.renorm*(mu + (A @ torch.exp(u).T).T)
        return ode


    ##########
    # defining the training sequence for the PINN

    def train(self, nb_epochs, theta, lambda_2=1.0, lambda_1=1.0, loss_1=True, verbose=True, loss_toler=1e-15):
        """Training sequence for the PINN

        Args:
            nb_epochs (int): number of epochs for the traininga
            lambda_2 (float, optional): Hyperparameter weighting the cost of the model. Defaults to 1.0
        """

        # lambda_2 is a parameter that weights the cost of the model
        # if 0 : only train on the data, if 1 : equal cost, if >1 more importance on the model
        lambda_2 = torch.tensor([lambda_2], requires_grad=False).float().to(self.device)
        lambda_1 = torch.tensor([lambda_1], requires_grad=False).float().to(self.device)

        mse_u = torch.tensor([0.0], requires_grad=False).float().to(self.device)
        mse_f = torch.tensor([0.0], requires_grad=False).float().to(self.device)
        mse_i = torch.tensor([0.0], requires_grad=False).float().to(self.device)

        # we iterate

        epoch = 0
        loss_item = 1e3

        mse_u_list = []
        mse_f_list = []


        while (epoch <= nb_epochs)*(loss_item >= loss_toler):
            # setting the grad to 0
            self.optimizer_Adam.zero_grad()

            # forward propagation
            U_pred = self.dnn(self.Tj)
            f_pred = self.functional_f(self.t_f, theta)

            # loss computing
            if loss_1:
                mse_u = torch.mean((U_pred - self.Uji)**2)
                mse_f = torch.mean((f_pred)**2)

                mse_f_list.append(mse_f.item())
                mse_u_list.append(mse_u.item())

                loss = mse_u + lambda_2 * mse_f
            else:
                if self.Tj.shape[0] == 1:
                    mse_i = torch.mean((U_pred - self.Uji)**2)
                    mse_f = torch.mean((f_pred)**2)

                    loss = mse_i + lambda_2 * mse_f

                else:
                    mse_u = torch.mean((U_pred[1:,:] - self.Uji[1:,:])**2)
                    mse_i = torch.mean((U_pred[0,:] - self.Uji[0,:])**2)
                    mse_f = torch.mean((f_pred)**2)

                loss = mse_i + lambda_2 * mse_f + lambda_1 * mse_u

            # backward propagation
            loss.backward()
            self.optimizer_Adam.step()

            # pbar.set_description(f'Loss {loss.item(): .3e}')
            # printing the loss at some iterations (each 1000 it.)
            if verbose and (epoch+1) % 1000 == 0:
                    print('Epoch: %d, Loss: %.3e' % ((epoch+1), loss.item()))

            epoch = epoch+1
            loss_item = loss.item()

        # plt.plot(mse_f_list, label='mse_f')
        # plt.plot(mse_u_list, label='mse_u')
        # plt.yscale('log')
        # plt.legend()
        # tikzplotlib.save("mse.pdf")
        # plt.show()

        if verbose :
            print('Epoch: %d, Loss: %.3e' % ((epoch+1), loss.item()))

        return {"mse_u": mse_u.item(), "mse_i": mse_i.item(), "mse_f": mse_f.item()}, [mse_u_list, mse_f_list]


    ##########
    # defining the prediction of the trained PINN

    def predict(self, t_eval):
        """Prediction of the trained PINN

        Args:
            t_eval (np.ndarray): time points where we want to evaluate the solution

        Returns:
            np.ndarray: solution at the time points t_eval
        """
        self.dnn.eval() # putting the model into evaluation mode

        #with torch.no_grad(): # getting rid of the gradient managment
        #    # setting the t_f
        #    t_f = torch.tensor(t_f, requires_grad=False).float().to(self.device)
        #    u_pred = self.dnn(t_f) # prediction
        #    u_pred = torch.exp(u_pred) # recovering the change of variable

        t_eval = torch.tensor(t_eval, requires_grad=True).float().to(self.device)
        u_pred = torch.exp(self.dnn(t_eval*(1/self.renorm)))
        return u_pred.detach().cpu().numpy()

    def predict_log_dt(self, t_eval):
        self.dnn.eval()
        t_eval = torch.tensor(t_eval, requires_grad=True).float().to(self.device)
        u_pred = self.dnn(t_eval*(1/self.renorm))
        u_t_pred = self.dt(u_pred, t_eval)
        return u_t_pred.detach().cpu().numpy()

    ##########
    # recovering the weigths of the DNN

    def get_weights(self):
        """Recovering the weights of the DNN

        Returns:
            dict: dictionary containing the weights of the DNN
        """
        # Call the get_weights method of the DNN class and return the weights as a dictionary
        return self.dnn.get_weights()


    ##########
    # Test performance of the PINN

    def test_performance(self, y0, theta, funode, name='PINN', plot=True, ntest=1000):
        """Test the performance of the PINN : compare the prediction of the PINN with the true solution of the GLV model

        Args:
            y0 (np.ndarray): initial population
            name (str, optional): Name to be displayed in the title. Defaults to 'PINN'.

        Returns:
            tuple: (t_online, err), where t_online is the time taken to compute the prediction and err is the relative error between the PINN and the true solution
        """
        
        t_end = self.t_f[-1][0].cpu()
        t_test = np.linspace(0, t_end, ntest)
        u_truth = odeint(funode, y0, t_test, args=(theta,), rtol=1e-7, atol=1e-9)

        t_begin = time.time()
        u_pinn = self.predict(t_test)
        t_end = time.time()

        err = np.linalg.norm(u_truth - u_pinn, 2, axis=0) / np.linalg.norm(u_truth, 2, axis=0)
        # print(f"Errors : {err}")
        # print(f"Online time = {t_end - t_begin:.5f} s")

        if plot:
            plt.plot(t_test, u_truth, label='u_truth')
            plt.gca().set_prop_cycle(None)
            plt.plot(t_test, u_pinn, '--', linewidth=2, label='u_pinn')
            plt.title(f"{name}\nFull: Truth, dashed: PINN")
            # plt.show()
            #plt.savefig(f"TrainLambda2-{name}.pdf")

        return t_end - t_begin, err
