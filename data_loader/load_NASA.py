import pandas as pd
import numpy as np
import torch

def normalize(X):
    """Normalize X to have mean 0 and std 1

    Parameters
    ----------
    X : `np.ndarray`, shape=(n, d)
        A time-independent features matrix

    Returns
    -------
    X_norm : `np.ndarray`, shape=(n, d)
        The corresponding normilized matrix with mean 0 and std 1
    """
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    X_norm = (X - mean) / std

    return X_norm

def get_NASA(nb_file):
    # get data from file and pre process it (normalization and convert to pandas)
    features_col_name = ['setting1', 'setting2', 'setting3'] + \
                        ["s" + str(i) for i in range(1, 22)]
    col_names = ['id', 'times'] + features_col_name
    dataset_train = pd.read_csv('data_loader/NASA/train_FD00{}.txt'.format(nb_file),
                                sep='\s+', header=None, names=col_names)
    dataset_test = pd.read_csv('data_loader/NASA/test_FD00{}.txt'.format(nb_file),
                               sep='\s+', header=None, names=col_names)

    dataset_train['tte'] = dataset_train.groupby(['id'])['times'].transform(max)
    dataset_train['label'] = 1
    dataset_test['tte'] = dataset_test.groupby(['id'])['times'].transform(max)
    dataset_test['label'] = 0
    dataset_test["id"] = dataset_test["id"] + max(dataset_train["id"])
    data = pd.concat([dataset_train, dataset_test])
    relevant_features_col_name = []
    for col in features_col_name:
        if not (len(dataset_train[col].unique()) < 10):
            relevant_features_col_name.append(col)
    data = data[["id", "times", "tte", "label"] + relevant_features_col_name]

    time_dep_features_col_name = relevant_features_col_name
    time_indep_features_col_name = []

    data[time_dep_features_col_name] = normalize(data[time_dep_features_col_name])
    data["times"] = data["times"] - 1
    data["tte"] = data["tte"] - 1

    return data, time_dep_features_col_name, time_indep_features_col_name


def load():
    df_org, cont_feat, bin_feat = get_NASA(1)
    df_org = df_org[~df_org.isin([np.nan, np.inf, -np.inf]).any(1)]
    time_scale = 100
    df_org["times"] = df_org["times"] / time_scale
    df_org["tte"] = df_org["tte"] / time_scale
    surv_times, surv_inds = tuple(df_org[["id", "tte", "label"]].drop_duplicates("id")[["tte", "label"]].values.T)
    # create sampling time grid is the combination of all longitudinal measurement time and survival time (start at 0)
    sampling_times = np.concatenate((np.zeros(1), np.unique(df_org[["times", "tte"]].values)))
    n_sampling_times = len(sampling_times)
    n_cont_feat = len(cont_feat)
    idxs = np.unique(df_org.id.values)
    n_samples = len(idxs)

    df_ = df_org[["id", "tte", "label", "times"] + cont_feat]
    X = np.zeros((n_samples, n_sampling_times, 1 + n_cont_feat), dtype=np.single)
    for i in np.arange(n_samples):
        idx = idxs[i]
        df_times = pd.DataFrame(np.array((np.array([idx] * n_sampling_times), sampling_times)).T, columns=["id", "times"])
        # merge with sampling time grid defined above, if individual i have no measurement at
        # a time point in this grid, fill it with the measurement value after this time point (priority).
        # If no measurement value after this time point, fill it with previous measurement value
        df_idx = pd.merge(df_times, df_[df_.id == idx], how="left", on=["id", "times"]).fillna(method='ffill').fillna(method='bfill')
        X[i] = df_idx[["times"] + cont_feat].values

    paths = torch.from_numpy(X.copy())
    surv_labels = np.array([surv_times, surv_inds], dtype=np.single).T

    # Dynamic-Deephit info for preprocessing
    bin_df = df_org[["id"] + bin_feat].drop_duplicates("id").sort_values(by=['id'])
    bin_df["id"] = np.arange(bin_df.shape[0])
    ddh_info_sup = (cont_feat, bin_feat, time_scale, bin_df)

    return paths, surv_labels, ddh_info_sup