import numpy as np


class CoxProx():
    """
    A class to define Cox model optimized by Proximal Gradient Descent

    Parameters
    ----------
    alpha :  `float`, default = 1e-2
        The penalty strength

    l1_ratio : `float`, default = 1e-1
        The ElasticNet mixing parameter.

    lr : `float`, default = 1e-3
        The initial learning rate at each iteration

    max_iter : `int`, default = 500
        The maximum iteration of learning optimizer
    """

    def __init__(self, alphas=1e-2, l1_ratio=1e-1, lr=1e-3, max_iters=500):

        self.alphas = alphas
        self.l1_ratio = l1_ratio
        self.lr = lr
        self.max_iters = max_iters

    def fit(self, feats, surv_ind, time_increment, n_samples):
        """
        Proximal gradient descent with back tracking linesearch


        Parameters
        ----------
        feats : `np.ndarray`, shape=(n_new_samples, coef_dim)
            The covariates

        surv_ind :  `np.ndarray`, shape=(n_pred_times)
            The survival indicator

        time_increment : `np.ndarray`, shape=(n_new_samples - n_samples, )
            The time increment array

        n_samples : `int`
            The number of samples (for scaling the likelihood)

        """
        coefs_dim = feats.shape[-1]
        current_coefs = np.zeros(coefs_dim)
        loss_values = []
        hist_coefs = []
        for iter in range(self.max_iters):
            lik_l1_pen, lik = self.likelihood(current_coefs, feats, surv_ind,
                                              time_increment, n_samples)
            loss_values.append(lik)
            hist_coefs.append(current_coefs)
            grad = self.grad(current_coefs, feats, surv_ind, time_increment, n_samples)
            # back-tracking linesearch
            for j in range(10):
                lr = self.lr * 10 ** (-j)
                next_coefs = self.prox(current_coefs, grad, lr)
                lik_l1_pen_, _ = self.likelihood(next_coefs, feats, surv_ind,
                                                 time_increment, n_samples)
                cond = lik_l1_pen + (grad * (next_coefs - current_coefs)).sum() + \
                       (1 / (2 * lr)) * ((next_coefs - current_coefs) ** 2).sum()

                if lik_l1_pen_ <= cond:
                    break

            current_coefs = next_coefs
        self.coefs = current_coefs
        self.hist_coefs = hist_coefs
        self.loss_values = loss_values

    def likelihood(self, current_coefs, feats, surv_ind, time_increment, n_samples):
        """
        Compute the negative log-likelihood  with penalty


        Parameters
        ----------
        current_coefs : `np.ndarray`, shape=(coef_dim, )
            The current coefficient

        feats : `np.ndarray`, shape=(n_new_samples, coef_dim)
            The covariates

        surv_ind :  `np.ndarray`, shape=(n_pred_times)
            The survival indicator

        time_increment : `np.ndarray`, shape=(n_new_samples - n_samples, )
            The time increment array

        n_samples : `int`
            The number of samples (for scaling the likelihood)

        Returns
        -------
        lik_l1_pen : `float`
            The negative log-likelihood  with L1 norm

        lik_pen : `float`
            The negative log-likelihood  with Elastic-Net
        """
        log_intensity = feats.dot(current_coefs)
        log_cum_intensity = np.exp(feats.dot(current_coefs)) * time_increment
        lik = - (1 / n_samples) * (log_intensity * surv_ind - log_cum_intensity).sum()
        l1_pen = self.alphas * self.l1_ratio * np.abs(current_coefs).sum()
        l2_pen = self.alphas * (1 - self.l1_ratio) * (current_coefs ** 2).sum()
        lik_l1_pen = lik + l1_pen
        lik_pen = lik + l1_pen + l2_pen

        return lik_l1_pen, lik_pen

    def prox(self, current_coefs, grad, lr):
        """
        Proximal operator of Lasso penalty (L1 norm)


        Parameters
        ----------
        current_coefs : `np.ndarray`, shape=(coef_dim, )
            The current coefficient

        grad : `np.ndarray`, shape=(coef_dim, )
            The gradient of negative log likelihood function at current coefficent

        lr :  `float`
            The learning rate

        Returns
        -------
        next_coefs : `np.ndarray`, shape=(coef_dim, )
            The updated coefficient

        """
        op1 = current_coefs - lr * grad
        op2 = lr * self.alphas * self.l1_ratio

        next_coefs = np.fmax(op1 - op2, 0) - np.fmax(-op1 - op2, 0)

        return next_coefs

    def grad(self, current_coefs, feats, surv_ind, time_increment, n_samples):
        """
        Compute the gradient of negative log-likelihood  with L2 norm


        Parameters
        ----------
        current_coefs : `np.ndarray`, shape=(coef_dim, )
            The current coefficient

        feats : `np.ndarray`, shape=(n_new_samples, coef_dim)
            The covariates

        surv_ind :  `np.ndarray`, shape=(n_pred_times)
            The survival indicator

        time_increment : `np.ndarray`, shape=(n_new_samples - n_samples, )
            The time increment array

        n_samples : `int`
            The number of samples (for scaling)

        Returns
        -------
        grad : `np.ndarray`, shape=(coef_dim, )
            The gradient

        """
        grad_lik = - (1 / n_samples) * (feats.T * (surv_ind -
                    np.exp(feats.dot(current_coefs)) * time_increment)).sum(axis=1)
        grad = grad_lik + 2 * self.alphas * (1 - self.l1_ratio) * current_coefs

        return grad

    def predict_hazard(self, sig):
        """
        Compute the intensity (hazard)


        Parameters
        ----------
        feats : `np.ndarray`, shape=(n_new_samples, coef_dim)
            The covariates

        Returns
        -------
        haz : `np.ndarray`, shape=(n_new_samples, )
            The predicted intensity

        """
        haz = np.exp(sig.dot(self.coefs))

        return haz