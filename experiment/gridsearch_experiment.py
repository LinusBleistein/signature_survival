import numpy as np
import time
import pickle as pkl
import sys
import pandas as pd
import itertools

from src.coxsig import CoxSignature
from competing_methods.coxfirst import CoxFirst
from competing_methods.rsf import RSF

sys.path.extend([".", ".."])

class GridSearchExp(object):
    """
    Base class, for hyper-parameters optimization experiments with hyperopt
    """

    def __init__(self, bst_name, output_folder_path="./", verbose=True):
        self.bst_name = bst_name
        self.output_folder_path = output_folder_path
        self.verbose = verbose
        self.default_params, self.best_params = None, None

        # to specify definitions in particular experiments
        self.space = None

    def optimize_params( self, X_train, y_train, X_val, y_val, pred_time,
                         eval_time, verbose=True, static_train=None, static_val=None):
        space = self.space
        list_param_names = list(space.keys())
        self.trials = pd.DataFrame(columns= list_param_names + ["result"])
        self.best_score = 1
        if len(list_param_names) == 1:
            space_ = list(map(tuple, list(space.values())[0].reshape(-1, 1)))
        elif len(list_param_names) == 2:
            p1, p2 = list(space.values())
            space_ = itertools.product(p1, p2)
        else:
            raise ValueError('The number hyperparameters exceed 2')
        for params in space_:
            result = self.run(X_train, y_train, X_val, y_val,
                              pred_time, eval_time, list(params),
                              verbose=verbose,static_train=static_train,
                              static_val=static_val)
            self.trials.loc[len(self.trials)] = list(params) + [result]
        self.best_results = self.trials["result"].min()
        best_trials = self.trials[self.trials.result == self.best_results].iloc[0]
        best_params = best_trials[list_param_names].values.tolist()
        self.best_params = self.preprocess_params(best_params)
        if self.verbose:
            filename = "best_params_results_" + str(self.bst_name) + ".pickle"

            with open(self.output_folder_path + filename, "wb") as f:
                pkl.dump(
                    {
                        "best_params": self.best_params,
                        "best_result": self.best_results,
                        "trials": self.trials,
                    }, f,
                )
        return self.best_params

    def run( self, X_train, y_train, X_val, y_val,
             pred_time, eval_time, params=None, verbose=False,
             static_train=None, static_val=None):
        print("params ", params)
        params = self.preprocess_params(params)
        start_time = time.time()
        self.fit(params, X_train, y_train, static_train)
        fit_time = time.time() - start_time
        cindex_score = np.mean(self.score(X_val, y_val, pred_time, eval_time, "c_index", static_val))
        bs_score = np.mean(self.score(X_val, y_val, pred_time, eval_time, "bs", static_val))
        score_ = - cindex_score + bs_score
        self.best_score = min(self.best_score, score_)

        if verbose:
            print(
                "[eval_time={0:.2f} sec\tcurrent_metric={1:.6f}\tmax_metric={2:.6f}".format(
                    fit_time,
                    score_,
                    self.best_score,
                )
            )
        return score_

    def fit(self, params, X_train, y_train, static_train):
        raise NotImplementedError("Method fit is not implemented.")

    def score(self, path, surv_label, pred_times, eval_times, metric, static_feat):
        score = self.learner.score(path, surv_label, pred_times, eval_times,
                                   metric, static_feat)
        return score

    def preprocess_params(self, params):
        raise NotImplementedError("Method preprocess_params is not implemented.")

class CoxFirstGridSearchExp(GridSearchExp):
    """
    Experiment class for CoxFirst, for hyper-parameters optimization
    experiments with hyperopt
    """

    def __init__(self, output_folder_path="./"):

        GridSearchExp.__init__(self, "CoxFirst", output_folder_path)
        # hard-coded params search space here
        self.space = {"alphas": .1**np.arange(0, 10)}

    def preprocess_params(self, params):

        params_ = {"alphas" : np.array(params)}

        return params_

    def fit(self, params, X_train, y_train, static_train=None):

        self.learner = CoxFirst(**params)
        self.learner.train(X_train, y_train, static_train)

class CoxSigGridSearchExp(GridSearchExp):
    """
    Experiment class for CoxFirst, for hyper-parameters optimization
    experiments with hyperopt
    """

    def __init__(self, output_folder_path="./"):

        GridSearchExp.__init__(self, "CoxSig", output_folder_path)
        # hard-coded params search space here
        self.space = {"sig_level" : [2, 3],
                      "alphas": .1**np.arange(0, 6)}

    def preprocess_params(self, params):

        params_ = {"sig_level" : int(params[0]),
                   "alphas" : params[1]}

        return params_

    def fit(self, params, X_train, y_train, static_train=None):

        self.learner = CoxSignature(**params)
        self.learner.train(X_train, y_train, static_train)

class CoxSigExtGridSearchExp(GridSearchExp):
    """
    Experiment class for CoxFirst, for hyper-parameters optimization experiments with hyperopt
    """

    def __init__(self, output_folder_path="./"):

        GridSearchExp.__init__(self, "CoxSigExt", output_folder_path)
        # hard-coded params search space here
        self.space = {"sig_level" : [2, 3],
                      "alphas": .1**np.arange(0, 6)}

    def preprocess_params(self, params):

        params_ = {"sig_level" : int(params[0]),
                   "alphas" : params[1]}

        return params_

    def fit(self, params, X_train, y_train, static_train=None):

        self.learner = CoxSignature(**params)
        self.learner.ext_ver = True
        self.learner.train(X_train, y_train, static_train)

class RSFGridSearchExp(GridSearchExp):
    """
    Experiment class for Random Survival Forest, for hyper-parameters
    optimization experiments with grid search
    """

    def __init__(self, output_folder_path="./", random_state=0):

        GridSearchExp.__init__(self, "RSF", output_folder_path)
        # hard-coded params search space here
        self.space = {"max_features": [None, "sqrt"],
                      "min_samples_leaf": [1, 5, 10]}
        self.random_state = random_state

    def preprocess_params(self, params):

        if params[0] is np.nan:
            params[0] = None
        params_ = {"max_features" : params[0],
                   "min_samples_leaf": int(params[1])}
        params_.update({"random_state": self.random_state})

        return params_

    def fit(self, params, X_train, y_train, static_train=None):

        self.learner = RSF(**params)
        self.learner.train(X_train, y_train, static_train)
        self.random_state += 1

def set_gridsearch_experiment(learner_name):

    experiment_setting = {
        "CoxFirst": CoxFirstGridSearchExp,
        "CoxSig": CoxSigGridSearchExp,
        "CoxSigExt": CoxSigExtGridSearchExp,
        "RSF": RSFGridSearchExp,
    }

    return experiment_setting[learner_name]

def run_gridsearch_exp(learner_name, paths, surv_labels, pred_time, eval_time,
                     output_folder_path, seed=0, nb_MC=5, run_hyperopt=True,
                       static_data = None):

    # Split train - test data
    train_test_share = 0.8
    n_samples = paths.shape[0]
    n_train_samples = int(train_test_share * n_samples)
    train_index = np.random.default_rng(seed).choice(n_samples, n_train_samples, replace=False)
    paths_train = paths[train_index, :, :]
    surv_labels_train = surv_labels[train_index, :]

    # NOTE: We do not need test set yet
    # Split tr - val data
    n_samples = paths_train.shape[0]
    tr_val_share = 0.8
    n_tr_samples = int(tr_val_share * n_samples)
    tr_index = np.random.default_rng(seed).choice(n_samples, n_tr_samples, replace=False)
    paths_tr = paths_train[tr_index, :, :]
    surv_labels_tr = surv_labels_train[tr_index, :]
    val_index = [i for i in np.arange(n_samples) if i not in tr_index]
    paths_val = paths_train[val_index, :, :]
    surv_labels_val = surv_labels_train[val_index, :]

    exp = set_gridsearch_experiment(learner_name)(output_folder_path)
    if learner_name == "RSF":
        exp.random_state = seed
    if run_hyperopt:
        tuned_params = exp.optimize_params(paths_tr, surv_labels_tr,
                                           paths_val, surv_labels_val,
                                           pred_time, eval_time)
    else:
        filename = "best_params_results_" + str(learner_name) + ".pickle"
        des_file = output_folder_path + filename
        with open(des_file, "rb") as f:
            f_ = pkl.load(f)
        tuned_params = f_["best_params"]

    seed_ = 0
    bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple = (), (), (), ()
    for i in range(nb_MC):
        n_samples = paths.shape[0]
        tr_te_share = 0.8
        n_tr_samples = int(tr_te_share * n_samples)
        tr_index = np.random.default_rng(seed_).choice(n_samples, n_tr_samples, replace=False)
        paths_tr = paths[tr_index, :, :]
        surv_labels_tr = surv_labels[tr_index, :]

        te_index = [i for i in np.arange(n_samples) if i not in tr_index]
        paths_te = paths[te_index, :, :]
        surv_labels_te = surv_labels[te_index, :]
        if static_data is not None:
            static_tr = static_data[tr_index]
            static_te = static_data[te_index]
        else:
            static_tr = None
            static_te = None
        exp.fit(tuned_params, paths_tr, surv_labels_tr, static_tr)
        bs = exp.score(paths_te, surv_labels_te, pred_time, eval_time, 'bs', static_te)
        cindex = exp.score(paths_te, surv_labels_te, pred_time, eval_time, 'c_index', static_te)
        w_bs = exp.score(paths_te, surv_labels_te, pred_time, eval_time, 'w_bs', static_te)
        auc = exp.score(paths_te, surv_labels_te, pred_time, eval_time, 'auc', static_te)
        bs_tuple += (bs,)
        cindex_tuple += (cindex,)
        w_bs_tuple += (w_bs,)
        auc_tuple += (auc,)
        seed_ += 1

    return bs_tuple, cindex_tuple, w_bs_tuple, auc_tuple