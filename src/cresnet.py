import torch
import torch.nn as nn
from torch import from_numpy
import numpy as np
from numpy.random import permutation

from src import vector_fields
from src.likelihood import get_log_likelihood
from src.utils import score, plot_loss_evolution, train_test_split


class ControlledResNet(nn.Module):
    """
    A class to define Neural Controlled Differential Equation, where the neural
    architecture is in type of ResNet

    Parameters
    ----------
    latent_dim : `int`
        Dimension of the latent state. (equal to the number of nodes in input layer)

    hidden_dim :  `int`
        The number of nodes in each hidden layer

    path_dim : `int`
        Dimension of the input time series.

    activation : `str`, default = tanh
        Activation function for all the nodes

    n_layers : `int`, default = 1
        Number of hidden layers (ResNet)

    sampling_times : `Tensor` or `np.ndarray` default = None
        The time channel of the input time series

    """

    def __init__(self, latent_dim, hidden_dim, path_dim, activation='tanh',
                 n_layers=1, sampling_times=None):

        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.path_dim = path_dim
        self.activation = activation
        self.n_layers = n_layers
        self.vector_field = vector_fields.VectorField(self.latent_dim,
                                                      self.hidden_dim,
                                                      self.path_dim,
                                                      self.activation,
                                                      self.n_layers)
        self.alpha = nn.Linear(self.latent_dim, 1)
        self.phi = nn.Linear(self.path_dim, self.latent_dim)
        if type(sampling_times) is np.ndarray:
            self.sampling_times = from_numpy(sampling_times)
        else:
            self.sampling_times = sampling_times

    def forward_latent(self, path):
        """
        Forward the current latent state through the Vector Field (ResNet),
        then multiply with the increment of input time series (path) and plus
        the current latent state. The result is the next (updated) latent state.


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        Returns
        -------
        latent_next : `np.ndarray`, shape=(n_samples, n_sampling_times, latent_dim)
            The predicted hazard function
        """
        path_init = path[:, 0, :]
        n_samples, n_sampling_times, _ = path.shape
        latent_init = self.phi(path_init).reshape(n_samples, -1)
        latent = latent_init.clone()
        latent_prev = latent_init.clone()
        for step in torch.arange(1, n_sampling_times):
            path_increment = (path[:, step, :] - path[:, step - 1, :]).flatten()
            latent_increment = (torch.block_diag(*self.vector_field(latent_prev)) @ path_increment)
            latent_increment = latent_increment.reshape(n_samples, -1)
            latent_current = latent_prev + latent_increment
            latent = torch.concat((latent, latent_current), 1)
            latent_prev = latent_current.clone()

        latent_next = latent.reshape(n_samples, n_sampling_times, -1)

        return latent_next

    def forward(self, path):
        """
        Feed the input time series (path) into Controlled Differential
        Equations where the Vector Field is modelled by neural network (ResNet).
        The result is the next latent state will be multiplied by a vector of
        coefficients (alpha) to have the intensity at the end.


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        Returns
        -------
        intensity : `np.ndarray`, shape=(n_samples, n_sampling_times)
            The intensity (hazard)
        """
        n_samples, n_sampling_times, _ = path.shape

        intensity = self.alpha(self.forward_latent(path))
        intensity = intensity.reshape(n_samples, n_sampling_times)

        return intensity

    def train(self, optimizer, path, surv_labels, batch_size, num_epochs,
              verbose = True, plot_loss=True, val_opt=False):
        """
        Train the Neural Controlled Differential Equation model


        Parameters
        ----------
        optimizer
            The specific optimizer for optimizing loss function

        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        surv_labels : `np.ndarray`, shape=(n_samples, 2)
            The survival label (survival time, censoring indicator)

        batch_size : `int`
            The number of sample in each batch (optimized the loss function in
            batched fashion)

        num_epochs : `int`, default = None
            The maximum number of training epochs

        verbose : `bool`= True, default = True
            Whether to print additional information during optimization.

        plot_loss : `bool`=True, default = True
            Whether to plot the loss function after optimization.

        val_opt : `bool`, default = False
            Whether use validation set in optimization.

        Returns
        -------
        intensity : `np.ndarray`, shape=(n_samples, n_sampling_times)
            The intensity (hazard)
        """
        self.surv_label_train = surv_labels
        if type(surv_labels) is np.ndarray:
            surv_labels = from_numpy(surv_labels)
        if val_opt:
            data_split = train_test_split(path, surv_labels, 0.8)
            paths_train, surv_labels_train, paths_val, surv_labels_val = data_split
        else:
            paths_train, surv_labels_train = path, surv_labels
        n_samples = paths_train.shape[0]
        num_batches = int(n_samples // batch_size)
        sampling_times = self.sampling_times
        epoch = 0
        loss_track = []
        loss_val_track = []
        seed = 0
        while epoch < num_epochs:
            epoch_loss = 0
            perm_idx = np.random.default_rng(seed).permutation(np.arange(n_samples))
            for i in range(num_batches):
                batch_idx = perm_idx[i * batch_size : (i + 1) * batch_size]
                paths_batch = paths_train[batch_idx]
                surv_labels_batch = surv_labels_train[batch_idx]
                log_hazard_batch = self.forward(paths_batch)
                batch_loss = - get_log_likelihood(log_hazard_batch,
                                                  sampling_times,
                                                  surv_labels_batch.long())
                batch_loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                with torch.no_grad():
                    epoch_loss += batch_loss.item()
            loss_track.append(epoch_loss)
            if verbose:
                print(f"epoch: {epoch}, objective: {epoch_loss}", end='\n')

            if val_opt:
                with torch.no_grad():
                    log_hazard_val = self.forward(paths_val)
                    loss_val = - get_log_likelihood(log_hazard_val,
                                                    sampling_times,
                                                    surv_labels_val.long()).item()
                    loss_val_track.append(loss_val)

            epoch += 1

        seed += 1
        loss_track = np.array(loss_track)
        self.loss = loss_track
        self.loss_val = np.array(loss_val_track)
        if plot_loss:
            plot_loss_evolution(loss_track, title = "NCDE loss over epoch",
                                xlabel = "Epoch", ylabel = "- Log-likelihood")

    def predict_hazard(self, path, pred_times):
        """
        Predict hazard function at all possible sampling times given
        longitudinal measurement up to prediction time


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        pred_times :  `np.ndarray`, shape=(n_pred_times)
            The prediction times

        Returns
        -------
        hazard : `np.ndarray`, shape=(n_samples, n_pred_times, n_sampling_times)
            The predicted hazard function
        """
        if type(pred_times) is np.ndarray:
            pred_times = from_numpy(pred_times)
        n_samples = path.shape[0]
        sampling_times = self.sampling_times
        eps = 1e-4
        n_sampling_times = len(sampling_times)
        n_pred_times = len(pred_times)
        hazard = torch.zeros((n_samples, n_pred_times, n_sampling_times))
        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            path_ = torch.clone(path)
            path_.swapaxes(0, 1)[j:, :, 1:] = path_[:, j, 1:]
            log_hazard = self.forward(path_)
            hazard[:, j] = torch.exp(log_hazard)

        return hazard

    def predict_survival(self, path, pred_times):
        """
        Predict conditional survival function evaluated from prediction times
        given the condition that the individual still alive at prediction time
        and all the longitudinal measurement up to this prediction time


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        pred_times :  `np.ndarray`, shape=(n_pred_times)
            The prediction times

        Returns
        -------
        cond_surv_preds : `np.ndarray`, shape=(n_samples, n_pred_times)
            The predicted survival function
        """
        if type(pred_times) is np.ndarray:
            pred_times = from_numpy(pred_times)
        hazard = self.predict_hazard(path, pred_times)
        sampling_times = self.sampling_times
        time_increment = sampling_times[1:] - sampling_times[:-1]
        cum_hazard = np.cumsum((hazard[:, :, :-1] * time_increment).detach().numpy(), axis=2)
        surv_preds = np.exp(-cum_hazard)

        cond_surv_preds = []
        n_pred_times = len(pred_times)
        eps = 1e-4
        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            time_pred_idx = np.searchsorted(sampling_times[1:], pred_time + eps)
            cond_surv_preds.append((surv_preds[:, j,
                                    time_pred_idx + 1:].T / surv_preds[:, j,
                                                            time_pred_idx]).T)

        return cond_surv_preds

    def score(self, path, surv_label, pred_times, eval_times, metric="bs"):
        """
        Predict conditional survival function evaluated from prediction times
        given the condition that the individual still alive at prediction time
        and all the longitudinal measurement up to this prediction time


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        surv_label :  `np.ndarray`, shape=(n_samples, 2)
            The survival label (survival time, censoring indicator)

        pred_times :  `np.ndarray`, shape=(n_pred_times)
            The prediction times

        eval_times :  `np.ndarray`, shape=(n_eval_times)
            The time windows need to evaluate the prediction performance
            after the prection time

        metric : `str`, default = "bs"
            The name of the metric need to be evaluated

        Returns
        -------
        scores : `np.ndarray`, shape=(n_pred_times, n_eval_times)
            Returns the scores in term of selected metrics.
        """
        n_samples = path.shape[0]
        n_pred_times = len(pred_times)
        n_eval_times = len(eval_times)
        sampling_times = self.sampling_times.detach().numpy()
        eps = 1e-4
        cond_surv_preds = self.predict_survival(path, pred_times)
        scores = np.zeros((n_pred_times, n_eval_times))

        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            surv_preds = np.zeros((n_samples, n_eval_times))
            for k in np.arange(n_eval_times):
                eval_time = eval_times[k]
                time_pred_idx = np.searchsorted(sampling_times[1:], pred_time + eps)
                time_eval_idx = np.searchsorted(sampling_times[1:][time_pred_idx + 1:], pred_time + eval_time + eps)
                surv_preds[:, k] = cond_surv_preds[j][:, time_eval_idx]

            # remove individuals whose survival time less than prediction time
            surv_times, surv_inds = surv_label[:, 0], surv_label[:, 1]
            idx_sel = surv_times >= pred_time
            surv_times_ = surv_times[idx_sel] - pred_time
            surv_inds_ = surv_inds[idx_sel]
            surv_labels_ = np.array([surv_times_, surv_inds_]).T
            surv_preds_ = surv_preds[idx_sel]

            scores[j] = score(metric, surv_labels_, surv_labels_,
                               surv_preds_, eval_times)

        return scores