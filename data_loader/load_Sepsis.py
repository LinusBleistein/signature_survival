import glob
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

def load():
    # Get a list of all the csv files
    csv_files = glob.glob('data_loader/Sepsis_extract/**.psv')

    # Statistic on missing level
    i = 0
    for file in csv_files:
        df = pd.read_csv(file, sep="|")
        is_all_missing = df.isnull().min()
        percent_missing = df.isnull().mean()
        if i == 0:
            all_missing_sample_df = pd.DataFrame(index=range(len(csv_files)),
                                                 columns=df.columns)
            missing_value_df = pd.DataFrame(index=range(len(csv_files)),
                                            columns=df.columns)
        missing_value_df.iloc[i] = percent_missing
        all_missing_sample_df.iloc[i] = is_all_missing
        i += 1

    missing_stat_df = pd.concat(
        [pd.DataFrame(all_missing_sample_df.mean(), columns=["all_missing"]),
         pd.DataFrame(missing_value_df.mean(), columns=["level_missing"])], axis=1)

    features_static_names = ["Age", "Gender"]
    unselected_features_static_names = ["Unit1", "Unit2", "HospAdmTime"]
    survival_indicator_names = ["SepsisLabel"]
    time_measurment_names = ["ICULOS"]

    features_convert_static = missing_stat_df[
        (~missing_stat_df.index.isin(unselected_features_static_names) &
         (missing_stat_df.all_missing <= .4) &
         (missing_stat_df.level_missing >= .5))].index.values.tolist()

    features_timedep_names = missing_stat_df[
        ((~missing_stat_df.index.isin(features_convert_static)) &
         ~missing_stat_df.index.isin(unselected_features_static_names) &
         ~missing_stat_df.index.isin(features_static_names) &
         ~missing_stat_df.index.isin(survival_indicator_names) &
         ~missing_stat_df.index.isin(time_measurment_names) &
         (missing_stat_df.all_missing < .4) &
         (missing_stat_df.level_missing < .5))].index.values.tolist()

    features_timeindep_names = features_static_names + features_convert_static

    # Handling missing data
    X, T, delta = [], [], []
    Y = pd.DataFrame([])
    idx = 0
    for file in csv_files:
        df = pd.read_csv(file, sep="|")
        feat = df[features_static_names
                  + features_convert_static
                  + features_timedep_names
                  + time_measurment_names
                  + survival_indicator_names]
        feat_drop_static_na = feat.dropna(subset=features_static_names)
        if not feat_drop_static_na.empty:
            # Handling missing values of converted static features by mean of each subject
            convert_static_feats_ = np.nanmean(feat_drop_static_na[features_convert_static].values, axis=0)
            static_feats_ = feat_drop_static_na[features_static_names].drop_duplicates().values.flatten()
            X.append(np.concatenate((convert_static_feats_, static_feats_)))

            survival_indicator = feat_drop_static_na[
                survival_indicator_names].values.flatten()
            if np.any(survival_indicator):
                T_i = min(feat_drop_static_na[feat_drop_static_na[survival_indicator_names].values.flatten() == 1][
                              time_measurment_names].values.flatten()) + 6
                Y_i = feat_drop_static_na[feat_drop_static_na[time_measurment_names].values.flatten() <= T_i][
                    features_timedep_names + time_measurment_names]
                delta_i = 1
            else:
                T_i = max(feat_drop_static_na[time_measurment_names].values.flatten())
                Y_i = feat_drop_static_na[features_timedep_names + time_measurment_names]
                delta_i = 0
            n_i = Y_i.shape[0]
            if Y_i.isnull().values.all(axis=0).sum():
                continue
            Y_i_ = np.hstack((idx * np.ones((n_i, 1)), Y_i.values))
            if Y.empty:
                columns = ["id"] + Y_i.columns.values.tolist()
                Y = pd.DataFrame(data=Y_i_, columns=columns)
            else:
                Y = Y.append(pd.DataFrame(data=Y_i_, columns=columns), ignore_index=True)
            T.append(T_i)
            delta.append(delta_i)
            idx += 1

    X = np.array(X)
    T = np.array(T)
    delta = np.array(delta)
    Y = Y.rename(columns={"ICULOS": "T_long"})

    id_list = np.unique(Y.id.values)
    id_non_censored = id_list[delta == 1]
    id_censored = id_list[delta == 0]

    nb_non_censored = len(id_non_censored)
    nb_selected_censored = int(.25 * nb_non_censored)
    id_selected_censored = np.random.default_rng(0).choice(id_censored, size=nb_selected_censored, replace=False)

    id_selected = np.sort(
        np.concatenate((id_non_censored, id_selected_censored))).astype('int')

    X = X[id_selected, :]
    T = T[id_selected]
    delta = delta[id_selected]
    Y = Y[Y.id.isin(list(id_selected))]

    # Handling missing values of converted static features by median of the whole data
    nan_inds = np.where(np.isnan(X))
    # Place column meadians in the indices. Align the arrays using take
    X[nan_inds] = np.take(np.nanmedian(X, axis=0), nan_inds[1])

    #sampling_times = np.unique(df_org[["times", "tte"]].values)
    time_scale = 100
    Y["T_long"] = (Y["T_long"] / time_scale).round(2)
    T = (T / time_scale).round(2)
    step = .01
    end_time = max(np.max(Y[["T_long"]].values), np.max(T)) + step
    sampling_times = np.arange(0, stop=end_time, step=step)
    n_sampling_times = len(sampling_times)

    longi_markers = features_timedep_names
    n_longi_markers = len(longi_markers)
    idxs = np.unique(Y.id.values)
    n_samples = len(idxs)

    df_ = Y[["id", "T_long"] + longi_markers]
    paths = np.zeros((n_samples, n_sampling_times, 1 + n_longi_markers), dtype=np.single)
    for i in np.arange(n_samples):
        idx = idxs[i]
        df_times = pd.DataFrame(np.array((np.array([idx] * n_sampling_times), sampling_times)).T, columns=["id", "T_long"])
        df_idx = pd.merge(df_times, df_[df_.id == idx], how="left", on=["id", "T_long"]).fillna(method='ffill').fillna(method='bfill')
        paths[i] = df_idx[["T_long"] + longi_markers].values


    paths = torch.from_numpy(paths.copy())
    sel_idx = paths.isnan().sum(axis=1).sum(axis=1) == 0
    paths = paths[sel_idx]
    sel_idx = np.array(sel_idx)

    surv_labels = np.array([T[sel_idx], delta[sel_idx]], dtype=np.single).T

    # dynamic deephit info for preprcessing
    bin_df = pd.DataFrame(data = np.hstack((np.arange(len(idxs[sel_idx])).reshape((-1, 1)),
                                            X[sel_idx])), columns=["id"] + features_timeindep_names)
    cont_feat = features_timedep_names
    bin_feat = features_timeindep_names
    ddh_info_sup = (cont_feat, bin_feat, time_scale, bin_df)

    return paths, surv_labels, ddh_info_sup, df_