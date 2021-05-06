'''
Copyright (c) 2020 AISyLab, TU Delft, The Netherlands. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

#!/usr/bin/env python

import tensorflow.keras as tk
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.utils import to_categorical

import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
from tensorflow.python.keras.utils import metrics_utils
from tensorflow.python.ops.losses import util as tf_losses_utils

import kerastuner as kt
from kerastuner.tuners import *
from kerastuner.engine.hypermodel import HyperModel
from kerastuner.engine.hyperparameters import HyperParameters

import sys
import h5py
import numpy as np
from scipy import stats
import scipy.stats as ss
import random
import math
from sklearn.preprocessing import StandardScaler

AES_Sbox = np.array([
    0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
    0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
    0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
    0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
    0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
    0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
    0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
    0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
    0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
    0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
    0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
    0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
    0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
    0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
    0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
    0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
])


def addGussianNoise(traces, noise_level):
    print('Add Gussian noise: ', noise_level)
    if noise_level == 0:
        return traces
    else:
        output_traces = np.zeros(np.shape(traces))
        for trace in range(len(traces)):
            if(trace % 5000 == 0):
                print(str(trace) + '/' + str(len(traces)))
            profile_trace = traces[trace]
            noise = np.random.normal(
                0, noise_level, size=np.shape(profile_trace))
            output_traces[trace] = profile_trace + noise
        return output_traces


def load_ascad(ascad_database_file, noise_level, load_metadata=False):
    try:
        in_file = h5py.File(ascad_database_file, "r")
    except:
        print("Error: can't open HDF5 file '%s' for reading (it might be malformed) ..." %
              ascad_database_file)
        sys.exit(-1)

    X_profiling = np.array(in_file['Profiling_traces/traces'], dtype=np.float64)
    X_profiling = addGussianNoise(X_profiling, noise_level)
    X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))

    Y_profiling = np.array(in_file['Profiling_traces/labels'])

    X_attack = np.array(in_file['Attack_traces/traces'], dtype=np.float64)
    X_attack = addGussianNoise(X_attack, noise_level)
    X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))

    Y_attack = np.array(in_file['Attack_traces/labels'])
    if load_metadata == False:
        return (X_profiling, Y_profiling), (X_attack, Y_attack)
    else:
        return (X_profiling[:50000], Y_profiling[:50000]), (X_attack[:10000], Y_attack[:10000]), (in_file['Profiling_traces/metadata']['plaintext'][:50000], in_file['Attack_traces/metadata']['plaintext'][:10000])


def labelize(plaintexts, keys):
    return AES_Sbox[plaintexts ^ keys]


def calculate_HW(data):
    hw = [bin(x).count("1") for x in range(256)]
    return [hw[s] for s in data]


def bit_diff(a, b):
    hw = [bin(x).count("1") for x in range(256)]
    container = np.zeros((len(a),))
    for i in range(len(a)):
        container[i] = hw[int(a[i]) ^ int(b[i])]
    return container


def calculate_MSB(data):
    if isinstance(data, (list, tuple, np.ndarray)):
        container = np.zeros((np.shape(data)), int)
        for i in range(len(data)):
            if data[i] >= 128:
                container[i] = 1
            else:
                container[i] = 0
    else:
        if data >= 128:
            container = 1
        else:
            container = 0
    return container

def calculate_LDD(k_c=34, mode='HW'):
    p = range(256)
    hw = [bin(x).count("1") for x in range(256)]
    k_all = range(256)
    container = np.zeros((len(k_all), len(p)), int)
    variance = np.zeros((256,))

    if mode == 'HW':
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = hw[labelize(p[i], k_all[j])]
        for k in range(256):
            variance[k] = np.sum(
                abs(np.power(container[k_c] - container[k], 2)))

    elif mode == 'ID':
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = labelize(p[i], k_all[j])
        for k in range(256):
            variance[k] = np.sum(abs(np.power(container[k_c] - container[k], 2)))

    else:
        for i in range(len(p)):
            for j in range(len(k_all)):
                container[j][i] = calculate_MSB(labelize(p[i], k_all[j]))
        for k in range(256):
            variance[k] = np.sum(abs(container[k_c] - container[k]))
    return variance


# Compute the evolution of rank
def rank_compute(prediction, att_plt, byte, output_rank):
    hw = [bin(x).count("1") for x in range(256)]
    (nb_traces, nb_hyp) = prediction.shape

    key_log_prob = np.zeros(256)
    prediction = np.log(prediction+1e-40)
    rank_evol = np.full(nb_traces,255)

    for i in range(nb_traces):
        for k in range(256):
            if leakage == 'ID':
                key_log_prob[k] += prediction[i, AES_Sbox[k ^ int(att_plt[i, byte])]]
            else:
                key_log_prob[k] += prediction[i, hw[AES_Sbox[k ^ int(att_plt[i, byte])]]]
        rank_evol[i] = rk_key(key_log_prob, correct_key)

    if output_rank:
        return rank_evol
    else:
        return key_log_prob


def perform_attacks(nb_traces, predictions, plt_attack, nb_attacks=1, byte=2, shuffle=True, output_rank=False):
    (nb_total, nb_hyp) = predictions.shape
    all_rk_evol = np.zeros((nb_attacks, nb_traces))

    for i in range(nb_attacks):
        if shuffle:
            l = list(zip(predictions, plt_attack))
            random.shuffle(l)
            sp, splt = list(zip(*l))
            sp = np.array(sp)
            splt = np.array(splt)
            att_pred = sp[:nb_traces]
            att_plt = splt[:nb_traces]

        else:
            att_pred = predictions[:nb_traces]
            att_plt = plt_attack[:nb_traces]

        key_log_prob = rank_compute(att_pred, att_plt, byte, output_rank)
        if output_rank:
            all_rk_evol[i] = key_log_prob

    if output_rank:
        return np.mean(all_rk_evol,axis=0)  
    else:
        return np.float32(key_log_prob)
        
def calculate_key_prob(y_true, y_pred):
    plt_attack = y_true[:, classes:]
    if plt_attack[0][0] == 1:  # check if data is from validation set, then compute GE
        GE = perform_attacks(nb_traces_attacks, y_pred, plt_attack[:, 1:], nb_attacks, byte=2)
    else:  # otherwise, return zeros
        GE = np.float32(np.zeros(256))
    return GE


@tf.function
def tf_calculate_key_prob(y_true, y_pred):
    _ret = tf.numpy_function(calculate_key_prob, [y_true, y_pred], tf.float32)
    return _ret

def calculate_rank(y_pred):
    pred_rank = ss.rankdata(y_pred, axis=1)-1
    return pred_rank/255

# Objective: GE
def rk_key(rank_array, key):
    key_val = rank_array[key]
    final_rank = np.float32(np.where(np.sort(rank_array)[::-1] == key_val)[0][0])
    if math.isnan(float(final_rank)) or math.isinf(float(final_rank)):
        return np.float32(256)
    else:
        return np.float32(final_rank)

# Objective: Lm
def calculate_Lm(key_prob):
    key_rank = 256-ss.rankdata(key_prob)
    corr, _ = stats.pearsonr(ranked_LDD, key_rank)
    if math.isnan(float(corr)) or math.isinf(float(corr)):
        return np.float32(0)
    else:
        return np.float32(corr)

def custom_loss(y_true, y_pred):
    return tk.backend.categorical_crossentropy(y_true[:, :classes], y_pred)

class acc_Metric(tk.metrics.Metric):
    def __init__(self, name='accuracy', **kwargs):
        super(acc_Metric, self).__init__(name=name, **kwargs)
        self.m = tf.keras.metrics.Mean()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.m.update_state(K.equal(K.argmax(y_true[:, :classes], axis=-1), K.argmax(y_pred, axis=-1)))

    def result(self):
        return self.m.result()

    def reset_states(self):
        self.m.reset_states()

class Lm_Metric(tk.metrics.Metric):
    def __init__(self, name='lm', **kwargs):
        super(Lm_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(calculate_Lm, [self.acc_sum], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))

class key_rank_Metric(tk.metrics.Metric):
    def __init__(self, name='key_rank', **kwargs):
        super(key_rank_Metric, self).__init__(name=name, **kwargs)
        self.acc_sum = self.add_weight(
            name='acc_sum', shape=(256), initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.acc_sum.assign_add(tf_calculate_key_prob(y_true, y_pred))

    def result(self):
        return tf.numpy_function(rk_key, [self.acc_sum, correct_key], tf.float32)

    def reset_states(self):
        self.acc_sum.assign(K.zeros(256))

def build_model_mlp(hp):
    activation = hp.Choice('activation', values=["relu", "tanh", "selu", "elu"])

    model = tk.models.Sequential()
    model.add(layers.Dense(units=hp.Int('dense_' + str(0) + '_units',
                                        min_value=100,
                                        max_value=400,
                                        step=100),
                           activation=activation,
                           input_shape=(input_length,)))

    for j in range(hp.Int('n_dense_layers', 1, 10)):
        model.add(layers.Dense(units=hp.Int('dense_' + str(j) + '_units',
                                            min_value=100,
                                            max_value=400,
                                            step=100),
                               activation=activation,
                               ))

    model.add(Dense(classes))
    model.add(Activation("softmax"))
    model.compile(optimizer=RMSprop(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])), loss=custom_loss, metrics=metric)

    model.summary()
    return model


def build_model(hp):
    activation = hp.Choice('activation', values=["relu", "tanh", "selu", "elu"])

    model = tk.models.Sequential()
    model.add(Conv1D(hp.Int('conv_0_filters', min_value=8, max_value=256, step=8),
                     hp.Int('conv_0_kernal_size', min_value=2, max_value=10, step=1), padding='same', input_shape=(input_length, 1)))
    model.add(Activation(activation))
    if hp.Choice('pooling_0', ['avg', 'max']) == 'max':
        model.add(MaxPooling1D(2, strides=2))
    else:
        model.add(AveragePooling1D(2, strides=2))

    for i in range(hp.Int('n_conv_layers', 1, 3)):
        model.add(Conv1D(hp.Int('conv_' + str(i + 1) + '_filters', min_value=8, max_value=256, step=8),
                         hp.Int('conv_' + str(i + 1) + '_kernal_size', min_value=2, max_value=14, step=1), padding='same'))
        model.add(Activation(activation))
        if hp.Choice('pooling_' + str(i + 1), ['avg', 'max']) == 'max':
            model.add(MaxPooling1D(hp.Int('maxPooling_' + str(i + 1) + '_size', min_value=2, max_value=5, step=1),
                                   hp.Int('maxPooling_' + str(i + 1) + '_stride', min_value=2, max_value=10, step=1)))
        else:
            model.add(AveragePooling1D(hp.Int('averagePooling_' + str(i + 1) + '_size', min_value=2, max_value=5, step=1),
                                       hp.Int('averagePooling_' + str(i + 1) + '_stride', min_value=2, max_value=10, step=1)))

    model.add(Flatten())
    for j in range(hp.Int('n_dense_layers', 1, 3)):
        model.add(layers.Dense(units=hp.Int('dense_' + str(j) + '_units', min_value=100, max_value=1600, step=100), activation=activation))

    model.add(Dense(classes))
    model.add(Activation("softmax"))
    model.compile(optimizer=RMSprop(lr=hp.Choice('learning_rate', values=[1e-3, 5e-4, 1e-4, 5e-5, 1e-5])), loss=custom_loss, metrics=metric)

    model.summary()
    return model


if __name__ == "__main__":
    root = 'root to the data'
    save_root = 'root the save the searching history'
    
    searching_method = 'BO'
    objective = 'val_lm' # 'val_lm/val_key_rank/val_acc'
    dataset = 'ASCAD'
    correct_key = 224
    leakage = 'HW'
    attack_model = 'MLP'
    noise_level = 0
    max_trails = 1
    naming_index = 'TEST'

    nb_traces_attacks = 2000
    nb_attacks = 3
    noise_level = 0

    if dataset == 'ASCAD':
        data_root = 'ASCAD/Base_desync0.h5'
    elif dataset == 'ASCAD_rand':
        data_root = 'ASCAD_rand/ascad-variable.h5'

    # load the data and normalize it
    (X_profiling, Y_profiling), (X_attack, Y_attack), (plt_profiling, plt_attack) = load_ascad(root + data_root, noise_level, load_metadata=True)
    input_length = len(X_profiling[0])
    scaler = StandardScaler()
    X_profiling = scaler.fit_transform(X_profiling)
    X_attack = scaler.transform(X_attack)

    if attack_model == 'MLP':
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1]))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1]))
        build_model = build_model_mlp
    else:
        X_profiling = X_profiling.reshape((X_profiling.shape[0], X_profiling.shape[1], 1))
        X_attack = X_attack.reshape((X_attack.shape[0], X_attack.shape[1], 1))
        build_model = build_model

    if leakage == 'ID':
        print('ID leakage model')
        classes = 256
    else:
        print('HW leakage model')
        classes = 9
        Y_profiling = calculate_HW(Y_profiling)
        Y_attack = calculate_HW(Y_attack)

    # calculate ideal key rank    
    ranked_LDD = ss.rankdata(calculate_LDD(correct_key, mode=leakage))

    # select the objective function
    if objective == 'val_accuracy':
        metric=[acc_Metric(), key_rank_Metric()] 
        direction = 'max'   
    elif objective == 'val_loss':
        direction = 'min'       
    elif objective == 'val_lm':
        metric=[Lm_Metric(), key_rank_Metric()] 
        direction = 'max'
    elif objective == 'val_key_rank':
        metric=[key_rank_Metric()] 
        direction = 'min'    
    else:
        print('No objective function defined!')

    Y_profiling = np.concatenate((to_categorical(Y_profiling, num_classes=classes), np.zeros((len(plt_profiling), 1)), plt_profiling), axis=1)
    Y_attack = np.concatenate((to_categorical(Y_attack, num_classes=classes), np.ones((len(plt_attack), 1)), plt_attack), axis=1)

    # select the searching method    
    if searching_method == 'BO':
        print('Searching method: BO')
        tuner = BayesianOptimization(build_model,
                                objective=kt.Objective(objective, direction=direction),
                                    max_trials=max_trails,
                                    executions_per_trial=1,
                                    directory=save_root,
                                    project_name=dataset + '_' + attack_model + '_' + leakage + '_'  + searching_method + '_' + objective + '_' + str(2),
                                    overwrite=True)
    else:
        print('Searching method: RA')
        tuner = RandomSearch(build_model,
                            objective=kt.Objective(objective, direction=direction),
                            max_trials=max_trails,
                            executions_per_trial=1,
                            directory=save_root,
                            project_name=dataset + '_' + attack_model + '_' + leakage + '_'  + searching_method + '_' + objective + '_' + str(2),
                            overwrite=True)

    tuner.search_space_summary()
    tuner.search(x=X_profiling, y=Y_profiling, epochs=10, batch_size=32, validation_data=(X_attack[:nb_traces_attacks], Y_attack[:nb_traces_attacks]), verbose=2)
    tuner.results_summary()

    # Retrain the best model
    print('Retrain the best model with 10 epochs...')
    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)
    model.fit(x=X_profiling, y=Y_profiling, batch_size=32, verbose=2, epochs=10)

    # Attack on the test traces with 10 epochs
    predictions = model.predict(X_attack)
    avg_rank = np.array(perform_attacks(5000, predictions, plt_attack, nb_attacks=10, byte=2, shuffle=True, output_rank=True))
    print('GE smaller that 1:', np.argmax(avg_rank < 1))
    print('GE smaller that 5:', np.argmax(avg_rank < 5))

    print('Retrain the best model with 50 epochs...')
    best_hp = tuner.get_best_hyperparameters()[0]
    model = tuner.hypermodel.build(best_hp)
    model.fit(x=X_profiling, y=Y_profiling, batch_size=32, verbose=2, epochs=50)

    # Attack on the test traces with 50 epochs
    predictions = model.predict(X_attack)
    avg_rank = np.array(perform_attacks(5000, predictions, plt_attack, nb_attacks=10, byte=2, shuffle=True, output_rank=True))
    print(np.shape(avg_rank))
    print('GE smaller that 1:', np.argmax(avg_rank < 1))
    print('GE smaller that 5:', np.argmax(avg_rank < 5))
