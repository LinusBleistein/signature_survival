import numpy as np
from sksurv.ensemble import RandomSurvivalForest
from src.utils import score, convert_surv_label_structarray

class RSF():

    def __init__(self, max_features = 'sqrt', min_samples_leaf = 10, random_state=0):
        self.model = RandomSurvivalForest(max_features=max_features,
                                          min_samples_leaf=min_samples_leaf,
                                          random_state=random_state)

    def train(self, path, surv_label, static_feat=None):
        self.surv_label_train = surv_label
        n_samples = path.shape[0]
        feat_matrix_ = np.array(path[:, 0, 1:]).reshape((n_samples, -1))
        if static_feat is not None:
            feat_matrix = np.concatenate((feat_matrix_, static_feat), axis=1)
        else:
            feat_matrix = feat_matrix_
        surv_time, surv_ind = surv_label[:, 0], surv_label[:, 1]
        surv_label_ext = np.array([surv_time, surv_ind]).T
        surv_label_ext_struct_array = convert_surv_label_structarray(surv_label_ext)

        self.model_fitted = self.model.fit(feat_matrix, surv_label_ext_struct_array)

    def predict_hazard(self, path, pred_times):
        # Not implemented
        pass

    def predict_survival(self, path, pred_times, static_feat=None):
        #TODO: Update with any predict time (predict time has to be in sampling times)
        n_samples = path.shape[0]
        feat_matrix_ = np.array(path[:, 0, 1:]).reshape((n_samples, -1))
        if static_feat is not None:
            feat_matrix = np.concatenate((feat_matrix_, static_feat), axis=1)
        else:
            feat_matrix = feat_matrix_
        surv_preds = self.model_fitted.predict_survival_function(feat_matrix)
        unique_times = self.model_fitted.event_times_

        cond_surv_preds = []
        n_pred_times = len(pred_times)
        eps = 1e-4
        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            time_pred_idx = np.searchsorted(unique_times[1:], pred_time + eps)
            cond_surv_preds.append((surv_preds[:, time_pred_idx + 1 : ].T / surv_preds[:, time_pred_idx]).T)

        return cond_surv_preds

    def score(self, path, surv_label, pred_times, eval_times, metric="bs", static_feat=None):
        n_samples = path.shape[0]
        n_pred_times = len(pred_times)
        n_eval_times = len(eval_times)
        unique_times = self.model_fitted.event_times_
        eps = 1e-4
        cond_surv_preds = self.predict_survival(path, pred_times, static_feat)
        results = np.zeros((n_pred_times, n_eval_times))

        for j in np.arange(n_pred_times):
            pred_time = pred_times[j]
            surv_preds = np.zeros((n_samples, n_eval_times))
            for k in np.arange(n_eval_times):
                eval_time = eval_times[k]
                time_pred_idx = np.searchsorted(unique_times[1:], pred_time + eps)
                time_eval_idx = np.searchsorted(unique_times[1:][time_pred_idx + 1:], pred_time + eval_time + eps)
                surv_preds[:, k] = cond_surv_preds[j][:, time_eval_idx]

            # remove individuals whose survival time less than prediction time
            surv_times, surv_inds = surv_label[:, 0], surv_label[:, 1]
            idx_sel = surv_times >= pred_time
            surv_times_ = surv_times[idx_sel] - pred_time
            surv_inds_ = surv_inds[idx_sel]
            surv_labels_ = np.array([surv_times_, surv_inds_]).T
            surv_preds_ = surv_preds[idx_sel]

            results[j] = score(metric, surv_labels_, surv_labels_,
                              surv_preds_, eval_times)

        return results