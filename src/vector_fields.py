import torch.nn as nn
import torch

class VectorField(nn.Module):
    """
    Class to create a learnable vector field. The vector field's forward takes
    as an input the latent CDE at some time point and outputs a matrix of size
    input_size*output_size.

    Parameters
    ----------

    latent_dim: `int`
        Dimension of the latent state.

    hidden_dim: `int`
        Size of the hidden layer of the vector field.

    path_dim: `int`
        Dimension of the input time series.

    activation: `str`
        Activation function for the field. Values can be "tanh" or "relu".

    n_layers: `int`
        Number of hidden layers.
    """

    def __init__(self, latent_dim, hidden_dim, path_dim, activation="tanh", n_layers=1):
        super().__init__()
        torch.manual_seed(0)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.path_dim = path_dim
        activations_dict = {"relu":nn.ReLU(), "tanh":nn.Tanh()}
        self.activation = activations_dict[activation]

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(latent_dim, hidden_dim))
        self.layers.append(self.activation)
        for i in range(n_layers):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(self.activation)
        self.layers.append(nn.Linear(hidden_dim, latent_dim * path_dim))

    def forward(self, z):
        """
        Feed forward the input through the network


        Parameters
        ----------
        z : `Tensor`, shape=(n_samples, latent_dim, path_dim)
            The latent state

        Returns
        -------
        output : `np.ndarray`, shape=(n_samples, n_sampling_times)
            The output
        """
        latent = z.clone()
        for layer in self.layers:
            latent = layer(latent)

        output = latent.reshape(-1, self.latent_dim, self.path_dim)

        return output

class RandomSigVectorField(nn.Module):
    """
    Class to create a randomized vector field. The vector field's forward takes
    as an input the latent CDE at some time point and outputs a matrix of size
    input_size*output_size.

    Parameters
    ----------

    latent_dim: `int`
        Dimension of the latent state.

    path_dim: `int`
        Dimension of the input time series.
    """
    def __init__(self, latent_dim, path_dim):
        super().__init__()
        self.latent_dim = latent_dim
        self.path_dim = path_dim
        self.A = nn.Linear(latent_dim, path_dim*latent_dim)
        self.scale = 1 / (path_dim * torch.sqrt(torch.tensor(latent_dim)))

        #Gaussian initialization, the dirty way
        self.A.weight = torch.nn.Parameter(torch.randn_like(self.A.weight), requires_grad=False)
        self.A.bias = torch.nn.Parameter(torch.randn_like(self.A.bias),requires_grad=False)

    def forward(self,z):
        """
        Feed forward the input through the network
        """

        return self.scale * self.A(z).reshape(-1, self.latent_dim, self.path_dim)
