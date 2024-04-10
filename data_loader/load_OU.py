import numpy as np
from src.datagen import Ornstein_Uhlenbeck

def load(n_samples = 500, n_sampling_times = 1000, end_time = 10, dim = 5,
         hurst = 0.6, intercept = True, threshold = 2.5):

    args = (n_samples, n_sampling_times, end_time, dim,
            hurst, intercept, threshold)
    simulated_dataset = Ornstein_Uhlenbeck().generate_simulated_dataset(*args)
    paths, sampling_times, surv_times, surv_inds = simulated_dataset
    surv_labels = np.array([surv_times, surv_inds]).T

    # dynamic deephit info for preprocessing
    time_scale = n_sampling_times / end_time
    cont_feat = ["X_" + str(i) for i in range(dim - 1)]
    bin_feat = []
    ddh_info_sup = (cont_feat, bin_feat, time_scale, [])

    return paths, surv_labels, ddh_info_sup