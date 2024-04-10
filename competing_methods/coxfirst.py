import numpy as np
from src.utils import score, convert_surv_label_structarray
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sklearn.preprocessing import StandardScaler

class CoxFirst():

    def __init__(self, alphas=np.array([0.5]), l1_ratio=0.1,
                 max_iter=100, verbose=False, fit_baseline_model=True):

        n_alphas = len(alphas)
        self.model = CoxnetSurvivalAnalysis(alphas=alphas,
                                            n_alphas=n_alphas,
                                            l1_ratio=l1_ratio,
                                            max_iter=max_iter,
                                            verbose=verbose,
                                            fit_baseline_model=fit_baseline_model
                                            )
        self.scaler = StandardScaler()

    def train(self, path, surv_label, static_feat=None):
        self.surv_label_train = surv_label
        # Take the first measurment as covariate (do not take into account time-dimesion)
        cox_feat_matrix = np.array(path[:, 0, 1:])
        if static_feat is not None:
            cox_feat_matrix_ext = np.concatenate((cox_feat_matrix, static_feat), axis=1)
        else:
            cox_feat_matrix_ext = cox_feat_matrix
        self.scaler.fit(cox_feat_matrix_ext)
        cox_feat_matrix_scale = self.scaler.transform(cox_feat_matrix_ext)

        cox_surv_time, cox_surv_ind = surv_label[:, 0], surv_label[:, 1]
        surv_label_ext = np.array([cox_surv_time, cox_surv_ind]).T
        cox_surv_label_ext_struct_array = convert_surv_label_structarray(surv_label_ext)
        self.model.fit(cox_feat_matrix_scale, cox_surv_label_ext_struct_array)
        self.unique_times = self.model._baseline_models[-1].cum_baseline_hazard_.x

    def get_baseline_hazard(self):
        unique_times = self.unique_times
        cum_baseline_haz = self.model._baseline_models[-1].cum_baseline_hazard_.y
        # Check unique time vs sampling time
        baseline_haz = np.zeros((len(unique_times)))
        baseline_haz[1:] = (cum_baseline_haz[1:] - cum_baseline_haz[:-1]) / (unique_times[1:] - unique_times[:-1])

        return baseline_haz

    def predict_hazard(self, path, static_feat=None):
        # TODO: Update with any predict time (predict time has to be in sampling times)
        n_samples = path.shape[0]
        cox_feat_matrix = np.array(path[:, 0, 1:])
        if static_feat is not None:
            cox_feat_matrix_ext = np.concatenate((cox_feat_matrix, static_feat), axis=1)
        else:
            cox_feat_matrix_ext = cox_feat_matrix
        self.scaler.fit(cox_feat_matrix_ext)
        cox_feat_matrix_scale = self.scaler.transform(cox_feat_matrix_ext)
        baseline_hazard = self.get_baseline_hazard()
        risk_score = np.exp(self.model.predict(cox_feat_matrix_scale)).reshape((n_samples, 1))
        haz = risk_score * baseline_hazard

        return haz

    def predict_survival(self, path, pred_times, static_feat=None):
        #TODO: Update with any predict time (predict time has to be in sampling times)
        hazard_pred = self.predict_hazard(path, static_feat)
        unique_times = self.unique_times
        time_increment = unique_times[1:] - unique_times[:-1]
        surv_preds = np.exp(-np.cumsum(hazard_pred[:, :-1] * time_increment, axis=1))

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
        unique_times = self.unique_times
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