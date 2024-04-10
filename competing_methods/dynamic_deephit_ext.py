_EPSILON = 1e-08

import numpy as np
import pandas as pd
import tensorflow as tf
import random
import os
import yaml
from src.utils import plot_loss_evolution
import competing_methods.Dynamic_DeepHit.import_data as impt
from competing_methods.Dynamic_DeepHit.class_DeepLongitudinal import Model_Longitudinal_Attention
from competing_methods.Dynamic_DeepHit.utils_helper import f_get_minibatch, f_get_boosted_trainset

import warnings
warnings.filterwarnings("ignore")

def construct_df(path, surv_labels, cont_feat, bin_feat = [], time_scale=1., bin_df=[]):
    # scale the time
    path[:, :, 0] = path[:, :, 0] * time_scale
    sampling_times = np.array(path[0, :, 0])
    surv_times = surv_labels[:, 0] * time_scale
    surv_inds = surv_labels[:, 1]
    n_samples = path.shape[0]
    data = []
    for i in range(n_samples):
        surv_times_i = surv_times[i]
        sel_idx_i = sampling_times <= surv_times_i
        times_i = sampling_times.copy()[sel_idx_i]
        n_sampling_times_i = len(times_i)
        ttes_i = [surv_times_i] * n_sampling_times_i
        labels_i = [surv_inds[i]] * n_sampling_times_i
        ids_i = [int(i)] * n_sampling_times_i
        X_i = path[i, :, 1:][sel_idx_i]
        data_i = np.concatenate((np.array([ids_i, ttes_i, times_i, labels_i]).T, X_i), axis=1)
        data.append(data_i)
    columns = ["id", "tte", "times", "label"] + cont_feat
    n_cols = len(columns)
    data = np.concatenate(data, axis=0)
    df = pd.DataFrame(data=data, columns=columns)
    df["id"] = df["id"].values.astype(int)
    if len(bin_feat) != 0:
        df = df.merge(bin_df, on="id")
    return df

def _f_get_pred(model, data, data_mi, pred_horizon):
    '''
        predictions based on the prediction time.
        create new_data and new_mask2 that are available previous or
        equal to the prediction time (no future measurements are used)
    '''
    new_data = np.zeros(np.shape(data))
    new_data_mi = np.zeros(np.shape(data_mi))

    meas_time = np.concatenate([np.zeros([np.shape(data)[0], 1]),
                                np.cumsum(data[:, :, 0], axis=1)[:, :-1]],
                               axis=1)

    for i in range(np.shape(data)[0]):
        last_meas = np.sum(meas_time[i, :] <= pred_horizon)

        new_data[i, :last_meas, :] = data[i, :last_meas, :]
        new_data_mi[i, :last_meas, :] = data_mi[i, :last_meas, :]

    return model.predict(new_data, new_data_mi)


def f_get_risk_predictions(model, data_, data_mi_, pred_time, eval_time):
    pred = _f_get_pred(model, data_[[0]], data_mi_[[0]], 0)
    _, num_Event, num_Category = np.shape(pred)

    risk_all = {}
    for k in range(num_Event):
        risk_all[k] = np.zeros(
            [np.shape(data_)[0], len(pred_time), len(eval_time)])

    for p, p_time in enumerate(pred_time):
        ### PREDICTION
        pred_horizon = int(p_time)
        pred = _f_get_pred(model, data_, data_mi_, pred_horizon)

        for t, t_time in enumerate(eval_time):
            eval_horizon = int(
                t_time) + pred_horizon  # if eval_horizon >= num_Category, output the maximum...

            # calculate F(t | x, Y, t >= t_M) = \sum_{t_M <= \tau < t} P(\tau | x, Y, \tau > t_M)
            risk = np.sum(pred[:, :, pred_horizon:(eval_horizon + 1)],
                          axis=2)  # risk score until eval_time
            risk = risk / (np.sum(np.sum(pred[:, :, pred_horizon:], axis=2), axis=1,
                                  keepdims=True) + _EPSILON)  # conditioniong on t > t_pred

            for k in range(num_Event):
                risk_all[k][:, p, t] = risk[:, k]

    return risk_all

class Dynamic_DeepHit_ext():
    def __init__(self, hparams=None):
        ##### HYPER-PARAMETERS
        if hparams is None:
            hparams_path = 'competing_methods/Dynamic_DeepHit/hparams.yaml'
            with open(hparams_path) as f:
                hparams = yaml.load(f, Loader=yaml.BaseLoader)
        self.burn_in_mode = 'ON'  # {'ON', 'OFF'}
        self.boost_mode = 'ON'  # {'ON', 'OFF'}

        # NETWORK HYPER-PARMETERS
        self.network_settings = {'h_dim_RNN': int(hparams['h_dim_RNN']),
                            'h_dim_FC': int(hparams['h_dim_FC']),
                            'num_layers_RNN': int(hparams['num_layers_RNN']),
                            'num_layers_ATT': int(hparams['num_layers_ATT']),
                            'num_layers_CS': int(hparams['num_layers_CS']),
                            'RNN_type': 'LSTM',
                            'FC_active_fn': tf.nn.relu,
                            'RNN_active_fn': tf.nn.tanh,
                            'initial_W': tf.contrib.layers.xavier_initializer(),

                            'reg_W': float(hparams['reg_W']),
                            'reg_W_out': float(hparams['reg_W_out'])
                            }

        self.mb_size = int(hparams['mb_size'])
        self.iteration = int(hparams['iteration'])
        self.iteration_burn_in = int(hparams['iteration_burn_in'])
        self.keep_prob = float(hparams['keep_prob'])
        self.lr_train = float(hparams['lr_train'])
        self.alpha = float(hparams['alpha'])
        self.beta = float(hparams['beta'])
        self.gamma = float(hparams['gamma'])

    def preprocess(self, df, cont_feat, bin_feat=[]):
        (x_dim, x_dim_cont, x_dim_bin), (data, time, label), \
        (mask1, mask2, mask3), (data_mi) = impt.import_dataset(df, bin_feat, cont_feat)

        _, num_Event, num_Category = np.shape(mask1)
        max_length = np.shape(data)[1]
        # INPUT DIMENSIONS
        self.input_dims = {'x_dim': x_dim,
                      'x_dim_cont': x_dim_cont,
                      'x_dim_bin': x_dim_bin,
                      'num_Event': num_Event,
                      'num_Category': num_Category,
                      'max_length': max_length}

        return (data, time, label), (mask1, mask2, mask3), (data_mi)

    def train(self, data_full, verbose=True, plot_loss=True, sess_path=None, is_trained=True):

        data, data_mi, label, time, mask1, mask2, mask3 = data_full

        ##### CREATE DYNAMIC-DEEPFHT NETWORK
        tf.reset_default_graph()

        if is_trained:
            sess = tf.Session()
            self.model = Model_Longitudinal_Attention(sess, "Dyanmic-DeepHit",
                                                      self.input_dims,
                                                      self.network_settings)
            saver = tf.train.Saver()
            saver.restore(sess, sess_path)
        else:
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            sess = tf.Session(config=config)
            self.model = Model_Longitudinal_Attention(sess, "Dyanmic-DeepHit",
                                                 self.input_dims, self.network_settings)
            saver = tf.train.Saver()
            sess.run(tf.global_variables_initializer())


            if self.boost_mode == 'ON':
                data, data_mi, label, time, mask1, mask2, mask3 = \
                    f_get_boosted_trainset(data, data_mi, label, time, mask1, mask2, mask3)
            loss_track = []
            ### TRAINING - BURN-IN
            if self.burn_in_mode == 'ON':
                if verbose:
                    print("BURN-IN TRAINING ...")
                for itr in range(self.iteration_burn_in):
                    x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(
                        self.mb_size, data, data_mi, label, time, mask1, mask2, mask3)
                    DATA = (x_mb, k_mb, t_mb)
                    MISSING = (x_mi_mb)

                    _, loss_curr = self.model.train_burn_in(DATA, MISSING,
                                                            self.keep_prob,
                                                            self.lr_train)
                    loss_track.append(loss_curr)
                    if verbose:
                        if (itr + 1) % 10 == 0:
                            print('itr: {:04d} | loss: {:.4f}'.format(itr + 1, loss_curr))

            ### TRAINING - MAIN
            if verbose:
                print("MAIN TRAINING ...")

            for itr in range(self.iteration):
                x_mb, x_mi_mb, k_mb, t_mb, m1_mb, m2_mb, m3_mb = f_get_minibatch(
                    self.mb_size, data, data_mi, label, time, mask1, mask2, mask3)
                DATA = (x_mb, k_mb, t_mb)
                MASK = (m1_mb, m2_mb, m3_mb)
                MISSING = (x_mi_mb)
                PARAMETERS = (self.alpha, self.beta, self.gamma)

                _, loss_curr = self.model.train(DATA, MASK, MISSING, PARAMETERS,
                                           self.keep_prob, self.lr_train)
                loss_track.append(loss_curr)
                if verbose:
                    if (itr + 1) % 10 == 0:
                        print('itr: {:04d} | loss: {:.4f}'.format(itr + 1, loss_curr))

            loss_track = np.array(loss_track)
            self.loss = loss_track
            if plot_loss:
                plot_loss_evolution(loss_track, title="Dynamic DeepHit loss over epoch",
                                    xlabel="Epoch", ylabel="Loss")

            saver.save(sess, sess_path)



    def predict(self, te_data, te_data_mi, pred_time, eval_time):
        risk_all = f_get_risk_predictions(self.model, te_data, te_data_mi,
                                          pred_time, eval_time)
        surv_preds = 1 - risk_all[0]

        return surv_preds

    def score(self, te_data, te_data_mi, pred_time, eval_time):
        pass
