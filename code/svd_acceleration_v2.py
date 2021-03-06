import json
import pickle
import joblib
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk

import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

"""
SVD acceleration using svd_classes_v2 classes.
Keras implementation hopefully means Keras-type speeds.
todo:
    apply pruning to all layers in aggregate
    calculate total weights lost
    

TensorFlow 2.5.0
TensorFlow Model Optimization 0.6.0 (for compatibility with TF 2.5.0)
"""
#%% load data
from load_preprocess import preprocess
# preprocess()
print("loading data...")
load_X_train = open("./pickles/X_train", 'rb')
load_y_train = open("./pickles/y_train", 'rb')
load_t_train = open("./pickles/t_train", 'rb')
load_X_test = open("./pickles/X_test", 'rb')
load_y_test = open("./pickles/y_test", 'rb')
load_t_test = open("./pickles/t_test", 'rb')


X_train = pickle.load(load_X_train)
y_train = pickle.load(load_y_train)
t_train = pickle.load(load_t_train)
X_test = pickle.load(load_X_test)
y_test = pickle.load(load_y_test)
t_test = pickle.load(load_t_test)

pin_scaler = joblib.load('./pickles/pin_scaler')
acc_scaler = joblib.load('./pickles/pin_scaler')

load_X_train.close()
load_y_train.close()
load_t_train.close()
load_X_test.close()
load_y_test.close()
load_t_test.close()
#%%
from svd_classes_v2 import make_LSTM_singular_model, PrunableTimeDistributed
model = keras.models.load_model("./model_saves/pretrained_sequential")
print("making singular value model...")
smodel = make_LSTM_singular_model(model)
#%%
import time
print("timing comparison of full and non-reduced singular models.")
# prediction time on the native keras implementation
start_time = time.perf_counter()
fy = model.predict(X_test)
print("full model timing: " + str(time.perf_counter() - start_time) + " sec")
start_time = time.perf_counter()
sy = smodel.predict(X_test)
print("singular model timing: " + str(time.perf_counter() - start_time) + " sec")
plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t_test[0], sy[0], label = "predicted pin location")
plt.plot(t_test[0], y_test[0], label = "actual pin location",alpha=.8)
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
# plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()
#%% pruning singular values to reduce model size
def split_train_random(batch_size, train_len):
    runs = X_train.shape[0]
    run_size = X_train.shape[1]
    indices = [(randint(1, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[index[0],index[1]:index[1]+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index[0],index[1]+train_len][0] for index in indices]))
    return X_mini, y_mini
#%% pruning
# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# def apply_pruning_to_LSTM(layer):
#     pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(
#         target_sparsity=.7, begin_step=0)
#     pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
#         initial_sparsity=0, final_sparsity=.8, begin_step=0, end_step=20000)
#     if not isinstance(layer, keras.layers.TimeDistributed):
#         return tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule)
#     return layer

# smodel = keras.models.clone_model(smodel, clone_function=apply_pruning_to_LSTM)
# smodel.compile(
#     loss="mse",
#     optimizer="adam",
#     metrics = ['accuracy']
# )

# X_mini, y_mini = split_train_random(32000, 100)

# smodel.fit(X_mini, y_mini, batch_size=32, validation_data=(X_test,y_test), epochs=20, callbacks=[pruning_callbacks.UpdatePruningStep()])

#%% regularization
# pruning was a bad idea
smodel = make_LSTM_singular_model(model, regularizer=True)

X_mini, y_mini = split_train_random(10000, 100)
smodel.compile(
    loss="mse",
    optimizer="adam"
)
smodel.fit(X_mini, y_mini, batch_size=32, validation_data=(X_test,y_test), epochs=10)
#%% analysis & plots
s = []
for layer in smodel.layers[:-1]:
    s.append(layer.cell.kernel.numpy())
    s.append(layer.cell.recurrent_kernel.numpy())

start_time = time.perf_counter()
sy = smodel.predict(X_test)
print("singular model timing: " + str(time.perf_counter() - start_time) + " sec")
plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t_test[0], sy[0], label = "predicted pin location")
plt.plot(t_test[0], y_test[0], label = "actual pin location",alpha=.8)
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
# plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()