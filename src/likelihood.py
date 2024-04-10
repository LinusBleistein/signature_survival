import torch

def get_log_survival(log_hazards, indicator, time_increment):
    """
    Compute the survival probability at time to event of individuals

    Parameters
    ----------
    log_hazards : `np.ndarray`, shape=(batch_size, n_sampling_times - 1,)
        Matrix of log intensities (computed by the model)

    indicator : `np.ndarray`, shape=(batch_size, n_sampling_times - 1, )
        Cross matrix to compare whether survival times are equal than
        sampling times

    time_increment : `np.ndarray`, shape=(n_sampling_times - 1, )
        Matrix of time increments

    Returns
    -------
    survival : `np.ndarray`, shape=(batch_size, )
        The survival probability at time to event of individuals

    """

    hazards = torch.exp(log_hazards)
    survival = (hazards * time_increment * indicator).sum(axis=1)

    return survival

def get_log_intensity(log_hazards, indicator):
    """
    Compute the log hazard rate (intensity) at time to event of individuals

    Parameters
    ----------
    log_hazards : `np.ndarray`, shape=(batch_size, n_sampling_times,)
        Matrix of log intensities (computed by the model)

    indicator : `np.ndarray`, shape=(batch_size, n_sampling_times, )
        Cross matrix to compare whether survival times are greater or equal than
        sampling times

    Returns
    -------
    log_intensity : `np.ndarray`, shape=(batch_size, )
        The log hazard rate (intensity) at time to event of individuals
    """

    log_intensity = (log_hazards * indicator).sum(axis=1)

    return log_intensity


def get_log_likelihood(log_hazards, sampling_times, surv_labels):
    """
    Compute the log-likelihood for a whole batch, return a scalar

    Parameters
    ----------
    log_hazards : `np.ndarray`, shape=(batch_size, n_sampling_times,)
        Matrix of log intensities (computed by the model)

    sampling_times  : `np.ndarray`, shape=(n_sampling_times, )
        The measurement time points

    surv_labels : `np.ndarray`, shape=(batch_size, 2)
        The survival label (survival time, censoring indicator)

    Returns
    -------
    likelihood : `np.ndarray`, shape=(batch_size, )
        The likelihood
    """

    n_samples = log_hazards.shape[0]
    surv_time, surv_ind = surv_labels[:, 0], surv_labels[:, 1]
    tte_indicator_1 = surv_time.reshape(-1, 1) == sampling_times
    tte_indicator_2 = surv_time.reshape(-1, 1) >= sampling_times[:-1]
    time_increment = sampling_times[1:] - sampling_times[:-1]

    log_intensity = get_log_intensity(log_hazards, tte_indicator_1)
    log_survival = get_log_survival(log_hazards[:, :-1], tte_indicator_2, time_increment)
    likelihood = (surv_ind * log_intensity - log_survival).sum() / n_samples

    return likelihood