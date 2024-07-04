import numpy as np
import torch
import pandas as pd
import os
import logging
import argparse
import pickle as pkl
import time

import warnings
warnings.filterwarnings('ignore')

from src.cresnet import ControlledResNet
from competing_methods.dynamic_deephit_ext import construct_df as ddh_construct_df
from competing_methods.dynamic_deephit_ext import Dynamic_DeepHit_ext
from src.utils import score
from gridsearch_experiment import run_gridsearch_exp
from data_loader import load_OU, load_NASA, load_Tumor_Growth
import matplotlib.pyplot as plt
import seaborn as sns

def ncde_exp(paths, surv_labels, pred_times, eval_times, seed=0, nb_MC=10):

    seed_ = seed
    bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple = (), (), (), ()
    for m in range(nb_MC):
        # Split train - test data
        train_test_share = 0.8
        n_samples = paths.shape[0]
        n_train_samples = int(train_test_share * n_samples)
        train_index = np.random.default_rng(seed_).choice(n_samples, n_train_samples, replace=False)
        test_index = [i for i in np.arange(n_samples) if i not in train_index]
        paths_train = paths[train_index, :, :]
        surv_labels_train = surv_labels[train_index, :]
        paths_test = paths[test_index, :, :]
        surv_labels_test = surv_labels[test_index, :]

        latent_dim = 4
        hidden_dim = 128
        path_dim = paths_train.shape[-1]
        activation = 'tanh'
        n_layers = 2
        learning_rate = 1e-4
        batch_size = 32
        num_epochs = 50
        sampling_times = np.array(paths_train[0, :, 0])
        model = ControlledResNet(latent_dim, hidden_dim, path_dim,
                                activation, n_layers, sampling_times)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        model.train(optimizer, paths_train, surv_labels_train,
                    batch_size, num_epochs, verbose=False, plot_loss=False)

        bs = model.score(paths_test, surv_labels_test, pred_times, eval_times)
        cindex = model.score(paths_test, surv_labels_test, pred_times, eval_times, 'c_index')
        w_bs = model.score(paths_test, surv_labels_test, pred_times, eval_times, "w_bs")
        auc = model.score(paths_test, surv_labels_test, pred_times, eval_times, 'auc')
        bs_tuple += (bs,)
        cindex_tuple += (cindex,)
        w_bs_tuple += (w_bs,)
        auc_tuple += (auc,)
        seed_ += 1
    return bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple

def dynamicdeephit_exp(paths, surv_labels, ddh_info_sup, pred_time, eval_time,
                       seed=0, nb_MC=5, sess_path=None, is_trained=True):

    seed_ = seed
    bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple = (), (), (), ()
    for m in range(nb_MC):
        # Split train - test data
        train_test_share = 0.8
        n_samples = paths.shape[0]
        n_train_samples = int(train_test_share * n_samples)
        train_index = np.random.default_rng(seed_).choice(n_samples, n_train_samples, replace=False)
        test_index = [i for i in np.arange(n_samples) if i not in train_index]

        # set-up for Dynamic DeepHit
        cont_feat, bin_feat, time_scale, bin_df = ddh_info_sup
        df = ddh_construct_df(paths.clone(), surv_labels, cont_feat, bin_feat, time_scale, bin_df)
        dynamic_deephit = Dynamic_DeepHit_ext()
        (data, time, label), (mask1, mask2, mask3), (data_mi) = \
            dynamic_deephit.preprocess(df, cont_feat, bin_feat)

        # split data
        tr_data, te_data = data[train_index, :, :], data[test_index, :, :]
        tr_data_mi, te_data_mi = data_mi[train_index, :, :], data_mi[test_index,
                                                             :, :]
        tr_time, te_time = time[train_index, :], time[test_index, :]
        tr_label, te_label = label[train_index, :], label[test_index, :]
        tr_mask1, te_mask1 = mask1[train_index, :, :], mask1[test_index, :, :]
        tr_mask2, te_mask2 = mask2[train_index, :, :], mask2[test_index, :, :]
        tr_mask3, te_mask3 = mask3[train_index, :], mask3[test_index, :]

        tr_data_full = (tr_data, tr_data_mi, tr_label, tr_time,
                        tr_mask1, tr_mask2, tr_mask3)

        # train
        out_path = sess_path + "model" + str(m)
        dynamic_deephit.train(tr_data_full, verbose=False, plot_loss=False,
                              sess_path=out_path, is_trained=is_trained)

        # predict
        pred_time_scale = pred_time * time_scale
        eval_time_scale = eval_time * time_scale
        surv_preds = dynamic_deephit.predict(te_data, te_data_mi,
                                             pred_time_scale, eval_time_scale)

        # score
        n_pred_times = len(pred_time)
        n_eval_times = len(eval_time)
        bs = np.zeros((n_pred_times, n_eval_times))
        cindex = np.zeros((n_pred_times, n_eval_times))
        auc = np.zeros((n_pred_times, n_eval_times))
        w_bs = np.zeros((n_pred_times, n_eval_times))
        surv_labels_test = surv_labels[test_index, :]
        for j in np.arange(n_pred_times):
            pred_time_ = pred_time[j]

            # remove individuals whose survival time less than prediction time
            surv_times = surv_labels_test[:, 0]
            surv_inds = surv_labels_test[:, 1]
            idx_sel = surv_times >= pred_time_
            surv_times_ = surv_times[idx_sel] - pred_time_
            surv_inds_ = surv_inds[idx_sel]
            surv_labels_ = np.array([surv_times_, surv_inds_]).T
            surv_preds_ = surv_preds[:, j][idx_sel]

            bs[j] = score("bs", surv_labels_, surv_labels_,
                          surv_preds_, eval_time)
            cindex[j] = score("c_index", surv_labels_, surv_labels_,
                              surv_preds_, eval_time)
            w_bs[j] = score("w_bs", surv_labels_, surv_labels_,
                          surv_preds_, eval_time)
            auc[j] = score("auc", surv_labels_, surv_labels_,
                              surv_preds_, eval_time)

        bs_tuple += (bs,)
        cindex_tuple += (cindex,)
        w_bs_tuple += (w_bs,)
        auc_tuple += (auc,)
        seed_ += 1

    return bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple

def set_dataloader(dataset_name):
    loaders_mapping = {
        "OU": load_OU,
        "Tumor_Growth": load_Tumor_Growth,
        "NASA": load_NASA
    }
    return loaders_mapping[dataset_name]

def run(learner_name, paths, surv_labels, pred_times, eval_times,
         ddh_info_sub=None, hyperopt_args=None, nb_MC=10, dataset_name=None):

    perf_df = pd.DataFrame(columns=["model", "p_t", "d_t", "bs", "c_index", "w_bs", "auc"])
    output_folder_path, seed, run_hyperopt = hyperopt_args
    if learner_name in ["CoxFirst", "CoxLast", "CoxSig", "CoxSigExt", "RSF"]:
        _, bin_feat, _, bin_df = ddh_info_sub
        # static_data = bin_df[bin_feat].values
        static_data = None
        bs, cindex, w_bs, auc = run_gridsearch_exp(learner_name,
                                                   paths, surv_labels,
                                                   pred_times, eval_times,
                                                   output_folder_path, seed,
                                                   nb_MC,
                                                   run_hyperopt, static_data)
    elif learner_name == "Dynamic_DeepHit":
        is_trained = not run_hyperopt
        bs, cindex, w_bs, auc  = dynamicdeephit_exp(paths, surv_labels, ddh_info_sub,
                                        pred_times, eval_times, seed, nb_MC,
                                                    sess_path = output_folder_path,
                                                    is_trained=is_trained)
    elif learner_name == "NCDE":
        bs, cindex, w_bs, auc  = ncde_exp(paths, surv_labels,
                              pred_times, eval_times, seed, nb_MC)
    else:
        raise NameError("Learner is not defined.")
    for j in range(len(pred_times)):
        for k in range(len(eval_times)):
            bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple = (), (), (), ()
            for m in range(nb_MC):
                bs_tuple += (bs[m][j, k],)
                cindex_tuple += (cindex[m][j, k],)
                w_bs_tuple += (w_bs[m][j, k],)
                auc_tuple += (auc[m][j, k],)
            list_row = [learner_name, pred_times[j], eval_times[k], bs_tuple,
                        cindex_tuple, w_bs_tuple, auc_tuple]
            perf_df.loc[len(perf_df)] = list_row

    return perf_df

def exp(learner_name, dataset_name, hyperopt_args=None, nb_MC=10, ):

    # load data
    loader = set_dataloader(dataset_name)
    # NOTE: ddh_info_sup for Dynamic DeepHit
    paths, surv_labels, ddh_info_sup = loader.load()
    # set pred_time
    tte = surv_labels[surv_labels[:, 1] == 1][:, 0]
    quantile_pred_times = np.array([.05, .1, .2])
    pred_times = np.quantile(np.array(tte), quantile_pred_times)
    n_eval_times = 3
    eval_times = []
    quantile_eval = .05
    for k in range(n_eval_times):
        quantile_eval_ext = quantile_pred_times + (k+1) * quantile_eval
        eval_time = np.quantile(np.array(tte), quantile_eval_ext)
        # The time window
        eval_times.append(max(eval_time - pred_times))
    eval_times = np.array(eval_times)
    nb_MC = 10
    perf_df = run(learner_name, paths, surv_labels, pred_times, eval_times,
                  ddh_info_sup, hyperopt_args, nb_MC, dataset_name)

    return perf_df


def plot_perf(perf_df, dataset, metric):
    prediction_times = np.unique(perf_df["p_t"].values)
    eval_times = np.unique(perf_df["d_t"].values)
    for p_t in prediction_times:
        for d_t in eval_times:
            # C_index
            cond = (perf_df.d_t == d_t) & (perf_df.p_t == p_t)
            res = perf_df[cond].explode(metric).dropna()
            sns.set(style='whitegrid', font="STIXGeneral", context='talk')

            f, ax = plt.subplots(figsize=(12, 7))
            [x.set_linewidth(2) for x in ax.spines.values()]
            [x.set_edgecolor('black') for x in ax.spines.values()]

            plot = sns.boxplot(y=metric, x="Model", data=res,
                               linewidth=3, saturation=1,
                               palette='colorblind', width=1,
                               gap=0.15, whis=0.8, linecolor="Black")

            ylim = 1.
            metric_title = metric

            if metric == "Brier Score":
                ylim = .5
                if (np.array(res[metric].values.tolist()) > .25).any():
                    ylim = .5
                if (np.array(res[metric].values.tolist()) > .5).any():
                    ylim = 1.
                metric_title = "BS"
            if metric == "Weighted Brier Score":
                ylim = .25
                if (np.array(res[metric].values.tolist()) > .25).any():
                    ylim = .5
                metric_title = "WBS"

            plot.set_ylim(0., ylim)
            _, xlabels = plt.xticks()
            _, ylabels = plt.yticks()
            ax.set_xticklabels(xlabels, size=25)
            ax.set_yticklabels(ylabels, size=18)
            ax.set_xlabel('')
            ax.set_ylabel(metric, size=30)

            props = dict(boxstyle='round', facecolor='wheat', alpha=1)

            # place a text box in upper left in axes coords
            if metric == "Brier Score":
                ax.text(0.02, .95,
                        '$t={:.2f}, \delta t = {:.2f}$'.format(p_t, d_t),
                        transform=ax.transAxes, fontsize=25,
                        verticalalignment='top', bbox=props)
            else:
                ax.text(0.02, .1,
                        '$t={:.2f}, \delta t = {:.2f}$'.format(p_t, d_t),
                        transform=ax.transAxes, fontsize=25,
                        verticalalignment='top', bbox=props)
            plt.savefig("updated_results/{}/{}_{}_pt={:.2f},dt={:.2f}.pdf".format(dataset, dataset, metric_title, p_t, d_t))
            plt.show()


if __name__ == "__main__":

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    random_state_seed = 0
    if not os.path.exists("results"):
        os.mkdir("results")
    results_home_path = "results/"

    datasets = ["OU", "Tumor_Growth", "Califrais", "NASA"]
    learner_names = ["CoxSig", "CoxSigExt", "NCDE", "CoxFirst", "RSF", "Dynamic_DeepHit"]


    for learner_name in learner_names:
        for dataset_name in datasets:

            logging.info("=" * 128)
            logging.info("Launching experiments for %s" % dataset_name + " " + learner_name)

            if not os.path.exists(results_home_path + dataset_name):
                os.mkdir(results_home_path + dataset_name)
            output_folder_path = results_home_path + dataset_name + "/"
            run_hyperopt_bool = True

            hyperopt_args = (output_folder_path, random_state_seed, run_hyperopt_bool)
            start_time = time.time()
            results = exp(learner_name, dataset_name, hyperopt_args)
            fit_time = time.time() - start_time

            filename = "exp_{}.pickle".format(str(learner_name))
            with open(output_folder_path + filename, "wb") as f:
                pkl.dump({"results": results, "times": fit_time,}, f,)

            logging.info("Saved results in file %s" % output_folder_path + filename)

    times_df = pd.DataFrame(columns=["Dataset", "Hyperopt", "Approach", "Learner", "Times"])
    perf_df_all = pd.DataFrame(columns=["Model", "p_t", "d_t",
                                        "Brier Score", "C-Index",
                                        "Weighted Brier Score", "AUC",
                                        "Dataset"])
    for dataset in datasets:
        for j in range(len(learner_names)):
            output_folder_path = "results/{}/".format(dataset)
            filename = "exp_{}.pickle".format(learner_names[j])
            file = open(output_folder_path + filename, "rb")
            res = pkl.load(file)
            perf_df_j = res["results"]
            perf_df_j["Dataset"] = [dataset] * perf_df_j.shape[0]
            perf_df_j.columns = ["Model",
                                 "p_t",
                                 "d_t",
                                 "Brier Score",
                                 "C-Index",
                                 "Weighted Brier Score",
                                 "AUC",
                                 "Dataset"]
            if j == 0:
                perf_df = perf_df_j
            else:
                perf_df = pd.concat([perf_df, perf_df_j])
        perf_df["p_t"] = perf_df["p_t"].round(5)
        perf_df["d_t"] = perf_df["d_t"].round(5)
        if (dataset == "NASA"):
            perf_df["p_t"] = (perf_df["p_t"] * 100).round(2)
            perf_df["d_t"] = (perf_df["d_t"] * 100).round(2)
        perf_df.columns = ["Model",
                           "p_t",
                           "d_t",
                           "Brier Score",
                           "C-Index",
                           "Weighted Brier Score",
                           "AUC",
                           "Dataset"]
        perf_df = perf_df.replace('Dynamic_DeepHit', 'DDH')
        perf_df = perf_df.replace('CoxFirst', 'Cox')
        perf_df = perf_df.replace('CoxSigExt', 'CoxSig+')
        perf_df = perf_df.replace('Surv_ODE', 'SLODE')

        perf_df_all = pd.concat([perf_df_all, perf_df])

    # Plot results
    for dataset in datasets:
        for metric in ["C-Index", "Brier Score", "Weighted Brier Score", "AUC"]:
            plot_perf(perf_df_all[perf_df_all.Dataset == dataset], dataset, metric)