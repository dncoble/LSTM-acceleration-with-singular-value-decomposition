import json
import pickle
import joblib
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk
import tensorflow_model_optimization as tfmot
"""
SVD acceleration using svd_classes_v2 classes.
Keras implementation hopefully means Keras-type speeds.

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
from svd_classes_v2 import make_LSTM_singular_model
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

prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# change this to polynomial decay
pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(0.8, 0)

def apply_pruning_to_LSTM(layer):
    if not isinstance(layer, keras.layers.TimeDistributed):
        print("lstm layer")
        return tfmot.sparsity.keras.prune_low_magnitude(layer)
    return layer

smodel = keras.models.clone_model(smodel, clone_function=apply_pruning_to_LSTM)
smodel.compile(
    loss="mse",
    optimizer="adam",
    metrics = ['accuracy']
)

X_mini, y_mini = split_train_random(10000, 100)
model.fit(X_mini, y_mini, validation_data=(X_test,y_test), epochs=20)
#%% analysis & plots
# from sklearn.metrics import mean_squared_error
# y_pred_scaled = pin_scaler.inverse_transform(model.predict(X_test).squeeze())
# y_test_scaled = pin_scaler.inverse_transform(y_test[0].squeeze())

# rmse = mean_squared_error(y_pred_scaled, y_test_scaled, squared=False)

# plt.figure(figsize=(7,3.3))
# plt.title("LSTM prediction of pin location")
# plt.plot(t_test[0], y_pred_scaled, label = "predicted pin location")
# plt.plot(t_test[0], y_test_scaled, label = "actual pin location",alpha=.8)
# plt.xlabel("time [s]")
# plt.ylabel("pin location [m]")
# plt.ylim((0.045, .23))
# plt.legend(loc=1)
# plt.tight_layout()
# plt.savefig("./plots/full_model_prediction.png", dpi=800)
# print("full model mean squared error: " + str(rmse))

# rmses = [None] * 20
# rmses[0] = rmse
# for i in range(len(reduced_y_pred)):
#     y_pred_scaled = pin_scaler.inverse_transform(reduced_y_pred[i])
#     rmse = mean_squared_error(y_pred_scaled, y_test_scaled, squared=False)
    
#     plt.figure(figsize=(7,3.3))
#     plt.title("LSTM prediction of pin location")
#     plt.plot(t_test[0], y_pred_scaled, label = "predicted pin location")
#     plt.plot(t_test[0], y_test_scaled, label = "actual pin location",alpha=.8)
#     plt.xlabel("time [s]")
#     plt.ylabel("pin location [m]")
#     plt.text(2, .20, "n - r = " + str(i), fontsize=15)
#     plt.ylim((0.045, .23))
#     plt.legend(loc=1)
#     plt.tight_layout()
#     plt.savefig("./plots/reduced_model_prediction_" + str(i + 1) + ".png", dpi=800)
#     rmses[i+1] = rmse

# # plot of RMSE
# ratio_rmse = [rmse/rmses[0] for rmse in rmses]

# plt.figure(figsize=(6, 4))
# plt.title("RMSE change with reduced rank")
# plt.plot(range(0, 20), ratio_rmse)
# plt.plot([0,19],[1,1], 'k--', label = 'unit ratio')
# plt.xlabel("n - r")
# plt.ylabel("RMSE(reduced)/RMSE(full)")
# plt.xlim((0, 19))
# plt.ylim((0.8, 2.0))
# plt.xticks(range(0, 20), [str(i) for i in range(0, 20)])
# plt.legend(loc=2)
# plt.tight_layout()
# plt.savefig("./plots/RMSE_plot.png", dpi=800)

# # plot of timing
# # ratio_timing = [reduced_t_/full_model_timing for reduced_t_ in reduced_t]
# # plt.figure(figsize=(6, 4))
# # plt.title("Timing change with reduced rank")
# # plt.plot(range(1, 20), ratio_timing)
# # plt.plot([1,19],[1,1], 'k--', label = 'unit ratio')
# # plt.xlabel("n - r")
# # plt.ylabel("timing(reduced)/timing(full)")
# # plt.xlim((1, 19))
# # plt.ylim((0.8, 2.0))
# # plt.xticks(range(1, 20), [str(i) for i in range(1, 20)])
# # plt.legend(loc=2)
# # plt.tight_layout()
# # plt.savefig("./plots/timing_plot.png", dpi=800)

# # a gif of predictions as n - r increases
# import imageio
# import os

# frame = 1
# with imageio.get_writer('./plots/reduce_rank.gif', mode='I', duration=0.25) as writer:
#     filename = "./plots/reduced_model_prediction_" + str(frame) + ".png"
#     while(os.path.exists(filename)):
#         image = imageio.imread(filename)
#         writer.append_data(image)
#         filename = "./plots/reduced_model_prediction_" + str(frame) + ".png"
#         frame += 1
#     writer.close()