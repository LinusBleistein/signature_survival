import numpy as np
import iisignature
import torch
from src.coxprox import CoxProx
from src.utils import score, plot_loss_evolution
from sklearn.preprocessing import StandardScaler

class CoxSignature():
    """
    A class to define signature-based Cox model

    Parameters
    ----------
    sig_level : `int`, default = 2
        The signature order

    alpha :  `float`, default = 1e-2
        The penalty strength

    l1_ratio : `float`, default = 1e-1
        The ElasticNet mixing parameter.

    lr : `float`, default = 1e-1
        Learning rate

    max_iter : `int`, default = 500
        The maximum iteration of learning optimizer

    plot_loss : `bool`` default = False
        Plot the loss evolution curve

    ext_ver : `bool`
        Indicator whether to train with extended version of Cox Signature

    """

    def __init__(self, sig_level=2, alphas=1e-2, l1_ratio= 1e-1, lr=1e-1,
                 max_iter=500, plot_loss=False, ext_ver=False):

        self.sig_level = sig_level
        self.model = CoxProx(alphas, l1_ratio, lr, max_iter)
        self.plot_loss = plot_loss
        self.scaler = StandardScaler()
        self.ext_ver = ext_ver

    def construct_cox_feats(self, path, surv_label, static_feat, pred_time=None):
        """
        Construct the required elements of Cox model with static features
        - Consider each individual at each sampling time as a new individual
        with the input feature is the mixed between the longitudinal measurements
        up to this sampling time and the static features and this new individual
        censors at this sampling time. If the sampling time is equal to survival time
        and the individual has experiented the event at this time, the new individual
        is also  has experiented the event at this time. All the returned outputs
        are considered as extension of inputs. The time that the extented version
        is computed up to, is survival time if the prection time is None, otherwise
        it is the prediction time.


        Parameters
        ----------
        path : `np.ndarray`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        surv_label :  `np.ndarray`, shape=(n_samples, 2)
            The survival label (survival time, censoring indicator)

        static_feat : `np.ndarray`, shape=(n_samples, static_dim)
            The static features

        pred_time : `float`, default = None
            The prediction time (the longitudinal feature will be masked after
            the prediction time)

        Returns
        -------
        path_ext : `Tensor`, shape=(n_new_samples, n_sampling_times, path_dim)
            The extention of the longitudinal features

        static_feat_ext : `np.ndarray`, shape=(n_new_samples, static_dim)
            The extention of the static features

        surv_ind_ext : `np.ndarray`, shape=(n_new_samples, )
            The extention of the survival indicator

        time_increment_ext : `np.ndarray`, shape=(n_new_samples - n_samples, )
            The array of all time increment between two consecutive measurment
            times of all new individual.

        """
        n_samples, n_sampling_times, path_dim = path.shape
        sampling_times = self.sampling_times
        surv_time = surv_label[:, 0]
        surv_ind = surv_label[:, 1]
        path_ext = []
        static_feat_ext = []
        surv_ind_ext = []
        time_increment = []
        if pred_time is None:
            pred_time = surv_time
        for i in range(n_samples):
            idx_pred_time = np.sum(sampling_times <= pred_time[i])
            sampling_times_i_ = sampling_times[sampling_times <= (surv_time[i])]
            time_increment.extend((sampling_times_i_[1:] - sampling_times_i_[:-1]).tolist())
            sampling_times_i = sampling_times[sampling_times <= (surv_time[i])][1:].tolist()
            n_sampling_times_i = len(sampling_times_i)
            if surv_ind[i] == 0:
                surv_ind_ext.extend([0] * (n_sampling_times_i))
            else:
                surv_ind_ext.extend([0] * (n_sampling_times_i - 1) + [1])
            for j in range(1, n_sampling_times_i + 1):
                if self.ext_ver:
                    if static_feat is not None:
                        first_longi_i = np.array(path[i, 0, 1:])
                        static_feat_i = np.concatenate((first_longi_i, static_feat[i]))
                        static_feat_ext.append(static_feat_i)
                    else:
                        first_longi_i = np.array(path[i, 0, 1:])
                        static_feat_ext.append(first_longi_i)
                else:
                    if static_feat is not None:
                        static_feat_ext.append(static_feat[i])

                path_ij_padding = torch.empty(1, n_sampling_times, path_dim)
                if sampling_times_i[j - 1] <= pred_time[i]:
                    path_ij_padding[:, :j + 1] = path[i, :j + 1]
                    path_ij_padding[:, j + 1:] = path[i, j]
                else:
                    path_ij_padding[:, :idx_pred_time + 1] = path[i, :idx_pred_time + 1]
                    path_ij_padding[:, idx_pred_time + 1:, 1:] = path[i, idx_pred_time, 1:]
                    k = np.sum(sampling_times <= sampling_times_i[j - 1])
                    path_ij_padding[:, idx_pred_time + 1:k, 0] = path[i, idx_pred_time + 1:k, 0]
                    path_ij_padding[:, k:, 0] = path[i, k - 1, 0]
                path_ext.append(path_ij_padding)
        path_ext = torch.cat(path_ext)

        # convert list to array
        static_feat_ext = np.array(static_feat_ext)
        surv_ind_ext = np.array(surv_ind_ext)
        time_increment = np.array(time_increment)

        return path_ext, static_feat_ext, surv_ind_ext, time_increment

    def train(self, path, surv_label, static_feat=None):
        """
        Train the Cox model with time-independent features optimzed by
        Proximal Gradient Descent

        - Extract extension version of survival data (if set ext_ver to True,
        add the first longitudinal measurement to the current static features)
        - Convert from extended longitudinal fearures to signaturebased features
        - Normalize the extended version of original static features
        - Optimized the Cox model by Proximal Gradient Descent


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        surv_label :  `np.ndarray`, shape=(n_samples, 2)
            The survival label (survival time, censoring indicator)

        static_feat : `np.ndarray`, shape=(n_samples, static_dim), default = None
            The static features
        """
        n_samples = path.shape[0]
        self.sampling_times = np.array(path[0, :, 0])
        required_feats = self.construct_cox_feats(path, surv_label, static_feat)
        path_ext, static_feat_ext, surv_ind_ext, time_increment = required_feats
        path_sig = iisignature.sig(path_ext, self.sig_level)
        if len(static_feat_ext) == 0:
            cox_covariates= path_sig
        else:
            self.scaler.fit(static_feat_ext)
            static_feat_ext_scale = self.scaler.transform(static_feat_ext)
            cox_covariates = np.concatenate((path_sig, static_feat_ext_scale), axis=1)

        self.model.fit(cox_covariates, surv_ind_ext, time_increment, n_samples)

        if self.plot_loss:
            loss_track = np.array(self.model.loss_values)
            plot_loss_evolution(loss_track, title="CoxSig Loss evolution",
                                xlabel="Iteration", ylabel="")

    def predict_hazard(self, path, pred_times, static_feat=None):
        """
        Predict hazard function at all possible sampling times given
        longitudinal measurement up to prediction time


        Parameters
        ----------
        path : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
            The longitudinal features

        pred_times :  `np.ndarray`, shape=(n_pred_times)
            The prediction times

        static_feat : `np.ndarray`, shape=(n_samples, static_dim), default = None
            The static features

        Returns
        -------
        haz : `np.ndarray`, shape=(n_samples, n_pred_times, n_sampling_times - 1)
            The predicted hazard function
        """
        n_samples = path.shape[0]
        n_pred_times = len(pred_times)
        sampling_times = self.sampling_times
        n_sampling_times = len(sampling_times)
        last_pred_time = sampling_times[-1]
        surv_label_fake = np.array([[last_pred_time] * n_samples, [0] * n_samples]).T
        haz = np.zeros((n_samples, n_pred_times, n_sampling_times - 1))
        for j in range(n_pred_times):
            pred_time = pred_times[j]
            pred_time_fake = [pred_time] * n_samples
            required_feats = self.construct_cox_feats(path, surv_label_fake,
                                                      static_feat, pred_time_fake)
            path_ext, static_feat_ext, _, _ = required_feats
            path_sig = iisignature.sig(path_ext, self.sig_level)
            if len(static_feat_ext) == 0:
                cox_covariates= path_sig
            else:
                self.scaler.fit(static_feat_ext)
                static_feat_ext_scale = self.scaler.transform(static_feat_ext)
                cox_covariates = np.concatenate((path_sig, static_feat_ext_scale), axis=1)
            haz[:, j] = self.model.predict_hazard(cox_covariates).reshape((n_samples, -1))

        return haz

    def predict_survival(self, path, pred_times, static_feat=None):
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

        static_feat : `np.ndarray`, shape=(n_samples, static_dim), default = None
            The static features

        Returns
        -------
        cond_surv_preds : `np.ndarray`, shape=(n_samples, n_pred_times)
            The predicted survival function
        """

        n_samples, n_sampling_times, _ = path.shape
        sampling_times = self.sampling_times
        n_pred_times = len(pred_times)

        hazard_pred = np.zeros((n_samples, n_pred_times, n_sampling_times))
        # hazard function at t=0 is assumed to be 0
        hazard_pred[:, :, 1:] = self.predict_hazard(path, pred_times, static_feat)
        time_increment = sampling_times[1:] - sampling_times[:-1]
        # survival function at t=0 is assumed to be 1
        surv_preds = np.exp(-np.cumsum(hazard_pred[:, :, :-1] * time_increment, axis=2))

        cond_surv_preds = []
        eps = 1e-4
        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            t_pred_id = np.searchsorted(sampling_times[1:], pred_time + eps)
            surv_pred_at_t = surv_preds[:, j, t_pred_id]
            cond_surv_pred = surv_preds[:, j, t_pred_id + 1 : ].T / surv_pred_at_t
            cond_surv_preds.append(cond_surv_pred.T)

        return cond_surv_preds

    def score(self, path, surv_label, pred_times, eval_times, metric="bs", static_feat=None):
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
            after the prediction time

        metric : `str`, default = "bs"
            The name of the metric need to be evaluated

        static_feat : `np.ndarray`, shape=(n_samples, static_dim), default = None
            The static features

        Returns
        -------
        scores : `np.ndarray`, shape=(n_pred_times, n_eval_times)
            Returns the scores in term of selected metrics.
        """

        n_samples = path.shape[0]
        n_pred_times = len(pred_times)
        n_eval_times = len(eval_times)
        sampling_times = self.sampling_times
        eps = 1e-4
        cond_surv_preds = self.predict_survival(path, pred_times, static_feat)
        scores = np.zeros((n_pred_times, n_eval_times))

        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            surv_preds = np.zeros((n_samples, n_eval_times))
            for k in np.arange(n_eval_times):
                eval_time = eval_times[k]
                t_pred_id = np.searchsorted(sampling_times[1:], pred_time + eps)
                time_eval_idx = np.searchsorted(sampling_times[1:][t_pred_id + 1:], pred_time + eval_time + eps)
                surv_preds[:, k] = cond_surv_preds[j][:, time_eval_idx]

            # remove individuals whose survival time less than prediction time
            surv_times, surv_inds = surv_label[:, 0], surv_label[:, 1]
            idx_sel = surv_times >= pred_time
            surv_times_ = surv_times[idx_sel] - pred_time
            surv_inds_ = surv_inds[idx_sel]
            surv_labels_ = np.array([surv_times_, surv_inds_]).T
            surv_preds_ = surv_preds[idx_sel]

            scores[j] = score(metric, surv_labels_, surv_labels_, surv_preds_, eval_times)

        return scores