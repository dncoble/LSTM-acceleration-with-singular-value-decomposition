import json
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk
"""
My final project for Linear Algebra, to reduce the size of this model with a
SVD-like method. But before I can do that I need to create my own
implementation of LSTM cells that I know I can use to evaluate timing fairly.
From there, the idea is to create a new model w/ units 30 -> 30 -> 20, perform
an SVD reduction on each matrix multiplication (prob half the rank?), and then
create another LSTM representation for timing. If all that works I really will
have done something.
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
#%%
from svd_classes import make_LSTM_Model, make_My_LSTM_Cell, make_Reduced_LSTM_Cell, My_LSTM_Model
import time
# prediction time on the native keras implementation
X_test_0 = np.expand_dims(X_test[0],axis=0)
start_time = time.monotonic()
model.predict(X_test_0)
print("native timing: " + str(time.monotonic() - start_time) + " sec")


my_model = make_LSTM_Model(model)
start_time = time.monotonic()
y_my_model = my_model.multi_step_forward_pass(X_test[0]) # 31.6 sec
full_model_timing = time.monotonic() - start_time
print("timing for full model: " + str(full_model_timing) + " sec")


# reduced models. I'll test reducing the rank of cells 2 and 3 from 1 to 15

# a python list that will contain numpy arrays for each reduction
reduced_y_pred = [None]*19
reduced_t = [None]*19
for i in range(1, 20):
    cell1 = make_My_LSTM_Cell(model.layers[0])
    cell2 = make_Reduced_LSTM_Cell(model.layers[1], 30-i)
    cell3 = make_Reduced_LSTM_Cell(model.layers[2], 20-i)
    reduced_model = My_LSTM_Model([cell1,cell2,cell3],model.layers[3])
    start_time = time.monotonic()
    reduced_y_pred[i-1] = reduced_model.multi_step_forward_pass(X_test[0])
    reduced_t[i-1] = time.monotonic() - start_time
    print("timing for reduced model: " + str(reduced_t[i-1]) + " sec")

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
ratio_timing = [reduced_t_/full_model_timing for reduced_t_ in reduced_t]
plt.figure(figsize=(6, 4))
plt.title("Timing change with reduced rank")
plt.plot(range(1, 20), ratio_timing)
plt.plot([1,19],[1,1], 'k--', label = 'unit ratio')
plt.xlabel("n - r")
plt.ylabel("timing(reduced)/timing(full)")
plt.xlim((1, 19))
plt.ylim((0.8, 2.0))
plt.xticks(range(1, 20), [str(i) for i in range(1, 20)])
plt.legend(loc=2)
plt.tight_layout()
plt.savefig("./plots/timing_plot.png", dpi=800)

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