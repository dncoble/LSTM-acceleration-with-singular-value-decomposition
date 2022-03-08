import json
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk
"""
Put a description of the work here
todo:
    replace units (30) with variable n
    replace amount of stacked cells with variable
    smarter drop metrics - sigma/weights, something with derivatives?
    
    
running on TensorFlow 2.5.0
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
model = keras.models.load_model("./model_saves/pretrained_sequential")
amplitude_model = keras.models.load_model('./model_saves/frequency')
#%%
from svd_classes import make_LSTM_Model, make_My_LSTM_Cell, make_Reduced_LSTM_Cell, My_LSTM_Model
from svd_classes import set_model_matrix_rank
from svd_classes import get_model_singular_values
import time
# prediction time on the native keras implementation
X_test_0 = np.expand_dims(X_test[0],axis=0)
start_time = time.monotonic()
model.predict(X_test_0)
print("native timing: " + str(time.monotonic() - start_time) + " sec")

# first index is the cell, second is matrix (W or U), third is gate (i,f,c,o)
model_ranks = np.full((4,2,4), 30)
model_singular_values = get_model_singular_values(model)

sorted_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(model_singular_values.ravel()), (4,2,4,30))))

# remove Ws of first layer
c = np.logical_and(sorted_indices[:,0] == 0, sorted_indices[:,1] == 0)
sorted_indices = sorted_indices[c == False]
#%% RMSE change with each
iterations =  sorted_indices.shape[0]- 20
rmse = np.zeros(iterations)
weights_eliminated = np.zeros(iterations)
running_weights = 0
y_test_scaled = pin_scaler.inverse_transform(y_test.squeeze())

from sklearn.metrics import mean_squared_error
from math import sqrt
for i in range(iterations):
    y_pred_scaled = pin_scaler.inverse_transform(model.predict(X_test).squeeze())
    mse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0]
    rmse[i] = sqrt(mse)
    weights_eliminated[i] = running_weights
    
    rank = model_ranks[sorted_indices[i][0],sorted_indices[i][1],sorted_indices[i][2]] - 1
    model_ranks[sorted_indices[i][0],sorted_indices[i][1],sorted_indices[i][2]] = rank
    model = set_model_matrix_rank(model, sorted_indices[i][:-1], rank)
    running_weights += 2*30 - 2*rank - 1
    print(str(i) + " out of " + str(iterations) + " reductions performed.")

ratio_rmse = [i/rmse[0] for i in rmse]

plt.figure(figsize=(6, 4))
plt.title("RMSE change with reduced rank")
plt.plot(weights_eliminated, ratio_rmse)
# plt.plot([0,19],[1,1], 'k--', label = 'unit ratio')
plt.xlabel("weights eliminated")
plt.ylabel("RMSE(reduced)/RMSE(full)")
# plt.xlim((0, 19))
# plt.ylim((0.8, 2.0))
# plt.xticks(range(0, 20), [str(i) for i in range(0, 20)])
# plt.legend(loc=2)
plt.tight_layout()
plt.savefig("./plots/RMSE_reducing_singular_values.png", dpi=800)


#%% analysis & plots
from sklearn.metrics import mean_squared_error
y_pred_scaled = pin_scaler.inverse_transform(y_my_model)
y_test_scaled = pin_scaler.inverse_transform(y_test[0].squeeze())

rmse = mean_squared_error(y_pred_scaled, y_test_scaled, squared=False)

plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t_test[0], y_pred_scaled, label = "predicted pin location")
plt.plot(t_test[0], y_test_scaled, label = "actual pin location",alpha=.8)
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig("./plots/full_model_prediction.png", dpi=800)
print("full model mean squared error: " + str(rmse))

rmses = [None] * 20
rmses[0] = rmse
for i in range(len(reduced_y_pred)):
    y_pred_scaled = pin_scaler.inverse_transform(reduced_y_pred[i])
    rmse = mean_squared_error(y_pred_scaled, y_test_scaled, squared=False)
    
    plt.figure(figsize=(7,3.3))
    plt.title("LSTM prediction of pin location")
    plt.plot(t_test[0], y_pred_scaled, label = "predicted pin location")
    plt.plot(t_test[0], y_test_scaled, label = "actual pin location",alpha=.8)
    plt.xlabel("time [s]")
    plt.ylabel("pin location [m]")
    plt.text(2, .20, "n - r = " + str(i), fontsize=15)
    plt.ylim((0.045, .23))
    plt.legend(loc=1)
    plt.tight_layout()
    plt.savefig("./plots/reduced_model_prediction_" + str(i + 1) + ".png", dpi=800)
    rmses[i+1] = rmse

# plot of RMSE
ratio_rmse = [rmse/rmses[0] for rmse in rmses]

plt.figure(figsize=(6, 4))
plt.title("RMSE change with reduced rank")
plt.plot(range(0, 20), ratio_rmse)
plt.plot([0,19],[1,1], 'k--', label = 'unit ratio')
plt.xlabel("n - r")
plt.ylabel("RMSE(reduced)/RMSE(full)")
plt.xlim((0, 19))
plt.ylim((0.8, 2.0))
plt.xticks(range(0, 20), [str(i) for i in range(0, 20)])
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("./plots/RMSE_plot.png", dpi=800)

# plot of timing
# ratio_timing = [reduced_t_/full_model_timing for reduced_t_ in reduced_t]
# plt.figure(figsize=(6, 4))
# plt.title("Timing change with reduced rank")
# plt.plot(range(1, 20), ratio_timing)
# plt.plot([1,19],[1,1], 'k--', label = 'unit ratio')
# plt.xlabel("n - r")
# plt.ylabel("timing(reduced)/timing(full)")
# plt.xlim((1, 19))
# plt.ylim((0.8, 2.0))
# plt.xticks(range(1, 20), [str(i) for i in range(1, 20)])
# plt.legend(loc=2)
# plt.tight_layout()
# plt.savefig("./plots/timing_plot.png", dpi=800)

# a gif of predictions as n - r increases
import imageio
import os

frame = 1
with imageio.get_writer('./plots/reduce_rank.gif', mode='I', duration=0.25) as writer:
    filename = "./plots/reduced_model_prediction_" + str(frame) + ".png"
    while(os.path.exists(filename)):
        image = imageio.imread(filename)
        writer.append_data(image)
        filename = "./plots/reduced_model_prediction_" + str(frame) + ".png"
        frame += 1
    writer.close()