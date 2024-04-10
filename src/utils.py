import numpy as np
import matplotlib.pyplot as plt
from sksurv.metrics import cumulative_dynamic_auc, brier_score

def plot_loss_evolution(loss_track, title, xlabel, ylabel):
    """
    Plot the loss curve

    Parameters
    ----------
    loss_track :  `np.ndarray`, shape=(n_samples, 2)
        Normal array of survival labels

    title : `str`
        Title of the figure

    xlabel : `str`
        Label of x axis

    ylabel : `str`
        Label of y axis
    """
    plt.figure(figsize=(8, 4))
    plt.plot(loss_track)
    plt.title(title, fontweight="bold")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


def convert_surv_label_structarray(surv_label):
    """
    Convert a normal array of survival labels to structured array. A structured
    array containing the binary event indicator as first field, and time of
    event or time of censoring as second field.

    Parameters
    ----------
    surv_label :  `np.ndarray`, shape=(n_samples, 2)
        Normal array of survival labels

    Returns
    -------
    surv_label_structarray : `np.ndarray`, shape=(n_samples, 2)
        Structured array of survival labels
    """
    surv_label_structarray = []
    n_samples = surv_label.shape[0]

    for i in range(n_samples):
        surv_label_structarray.append((bool(surv_label[i, 1]), surv_label[i, 0]))

    surv_label_structarray = np.rec.array(surv_label_structarray,
                                          dtype=[('indicator', bool),
                                                 ('time', np.float32)])

    return surv_label_structarray

def score(metric, survival_train, survival_test, estimates, times):
    """
    Score the prediction performance given the survival function estimation at
    evaluation time.


    Parameters
    ----------
    metric : `str`
        The selected metric. Values can be "bs", "w_bs", "c_index" or "auc".

    survival_train :  `np.ndarray`, shape=(n_samples, 2)
        Survival labels for training data to estimate the censoring
        distribution from.

    survival_test :  `np.ndarray`, shape=(n_samples, 2)
        Survival labels of testing data.

    estimates :  `np.ndarray`, shape=(n_eval_times)
        Estimated probability of remaining event-free at time points specified
        by times.

    times :  `np.ndarray`, shape=(n_times)
        The time points for which to estimate the score

    Returns
    -------
    scores : `np.ndarray`, shape=(n_times)
        Returns the scores in term of selected metric.
    """

    survival_train_struct_arr = convert_surv_label_structarray(survival_train)
    survival_test_struct_arr = convert_surv_label_structarray(survival_test)
    n_eval_time = len(times)

    if metric == "bs":
        surv_time, surv_ind = survival_test[:, 0], survival_test[:, 1]
        results = np.zeros(n_eval_time)
        for k in np.arange(n_eval_time):
            eval_time = times[k]
            at_risk = surv_time > eval_time
            results[k] = np.mean(at_risk * (1 - estimates[:, k])**2 +
                                 ~at_risk * surv_ind * estimates[:, k]**2)


    elif metric == "w_bs":
        # Consider the weight of censoring
        results = np.zeros(n_eval_time)
        for k in range(n_eval_time):
            estimate = estimates[:, k]
            eval_time = times[k]
            try:
                results[k] = brier_score(survival_train_struct_arr,
                                         survival_test_struct_arr,
                                         estimate, eval_time)[1]
            except ValueError:
                results[k] = np.nan


    elif metric == "auc":
        results = np.zeros(n_eval_time)
        for k in range(n_eval_time):
            eval_time = times[k]
            try:
                results[k], _ = cumulative_dynamic_auc(survival_train_struct_arr,
                                                       survival_test_struct_arr,
                                                       -estimates[:, k], eval_time)
            except ValueError:
                results[k] = np.nan


    elif metric == "c_index":
        surv_time, surv_ind = survival_test[:, 0], survival_test[:, 1]
        n_samples = estimates.shape[0]
        results = np.zeros(n_eval_time)

        for k in np.arange(n_eval_time):
            eval_time = times[k]

            A = np.zeros((n_samples, n_samples))
            Q = np.zeros((n_samples, n_samples))
            N_t = np.zeros((n_samples, n_samples))
            for i in range(n_samples):
                A[i, np.where(surv_time[i] < surv_time)] = 1
                Q[i, np.where(-estimates[i, k] > -estimates[:, k])] = 1

                if ((surv_time[i] <= eval_time) and (surv_ind[i] == 1)):
                    N_t[i, :] = 1

            Num = np.sum(((A) * N_t) * Q)
            Den = np.sum((A) * N_t)

            if Num == 0 and Den == 0:
                results[k] = np.nan
            else:
                results[k] = float(Num / Den)

    else:
        raise Exception("Unsupported metric")

    return results


def train_test_split(paths, surv_labels, train_test_share=.8):
    """
    Split the survival data into train set and test set with specific ratio.


    Parameters
    ----------
    paths : `Tensor`, shape=(n_samples, n_sampling_times, path_dim)
        The longitudinal features

    surv_labels :  `np.ndarray`, shape=(n_samples, 2)
        The survival label (survival time, censoring indicator)

    train_test_share :  `float`
        Split ratio

    Returns
    -------
    paths_train: `Tensor`, shape=(n_train_samples, n_sampling_times, path_dim)
        The training longitudinal features

    surv_labels_train :  `np.ndarray`, shape=(n_train_samples, 2)
        The training survival label (survival time, censoring indicator)

    paths_test: `Tensor`, shape=(n_test_samples, n_sampling_times, path_dim)
        The testing longitudinal features

    surv_labels_test :  `np.ndarray`, shape=(n_test_samples, 2)
        The testing survival label (survival time, censoring indicator)
    """
    n_samples = paths.shape[0]
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.choice(n_samples, n_train_samples, replace=False)
    test_index = [i for i in np.arange(n_samples) if i not in train_index]

    paths_train = paths[train_index, :, :]
    surv_labels_train = surv_labels[train_index, :]

    paths_test = paths[test_index, :, :]
    surv_labels_test = surv_labels[test_index, :]

    return paths_train, surv_labels_train, paths_test, surv_labels_test