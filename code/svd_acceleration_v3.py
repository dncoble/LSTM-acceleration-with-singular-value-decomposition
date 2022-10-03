import math
import numpy as np
from numpy.random import randint
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk
import time
from scipy import signal
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.sparsity.keras import prune
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_callbacks
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_schedule
from tensorflow_model_optimization.python.core.sparsity.keras import pruning_wrapper

from svd_classes_v3 import make_LSTM_singular_model, PrunableTimeDistributed, make_LSTM_reduced_model

"""
SVD acceleration using svd_classes_v3 classes. Now using TF 2.10 for built-in
orthogonal regularizer.

TensorFlow 2.10.0
"""
#%% load data
def preprocess(sampling_period):
    import json
    import pickle
    import numpy as np
    import sklearn as sk
    import joblib
    f = open('data_6_with_FFT.json')
    data = json.load(f)
    f.close()
    
    acc = np.array(data['acceleration_data'])
    acc_t = np.array(data['time_acceleration_data'])
    pin = np.array(data['measured_pin_location'])
    pin_t = np.array(data['measured_pin_location_tt'])
    
    # pin contains some nan values
    from math import isnan
    for i in range(len(pin)):
        if(isnan(pin[i])):
            pin[i] = pin[i-1]
    
    resample_period = sampling_period
    pin = pin[pin_t > 1.5]
    pin_t = pin_t[pin_t > 1.5] - 1.5
    acc = acc[acc_t > 1.5]
    acc_t = acc_t[acc_t > 1.5] - 1.5
    num = int((acc_t[-1] - acc_t[0])/resample_period)
    
    resample_acc, resample_t = signal.resample(acc, num, acc_t)
    resample_pin = np.interp(resample_t, pin_t, pin)
    
    # scaling data, which means that it must be unscaled to be useful
    from sklearn import preprocessing
    acc_scaler = sk.preprocessing.StandardScaler()
    acc_scaler.fit(resample_acc.reshape(-1, 1))
    acc = acc_scaler.fit_transform(resample_acc.reshape(-1, 1)).flatten()
    pin_scaler = sk.preprocessing.StandardScaler()
    pin_scaler.fit(resample_pin.reshape(-1,1))
    pin = pin_scaler.fit_transform(resample_pin.reshape(-1,1)).flatten().astype(np.float32)
    
    # reshape for multi-input
    ds = 16
    X = np.reshape(acc[:acc.size//ds*ds], (acc.size//ds, ds))
    t = np.reshape(resample_t[:resample_t.size//ds*ds], (resample_t.size//ds, ds)).T[0]
    y = np.reshape(pin[:pin.size//ds*ds], (pin.size//ds, ds)).T[0]
    
    X = np.expand_dims(X, 0)
    
    X_train = X[:,t<30.7]
    y_train = y[t<30.7]
    t_train = t[t<30.7]
    
    X_test = X[:,t>30.7]
    y_test = y[t>30.7]
    t_test = t[t>30.7]
    
    return (X, X_train, X_test), (y, y_train, y_test), (t, t_test, t_train), pin_scaler, acc_scaler

def split_train_random(X_train, y_train, batch_size, train_len):
    run_size = X_train.shape[1]
    indices = [randint(0, run_size - train_len) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[0,index:index+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index+train_len] for index in indices]))
    return X_mini, y_mini

# use the formula SNR= (A_signal/A_noise)_rms^2. returned in dB
def signaltonoise(signal, noisy_signal, invert=False, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    if(not invert):
        snr = (a_sig/a_noise)**2
    else:
        snr = (a_noise/a_sig)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
def apply_pruning_to_LSTM(layer):
    # pruning_schedule = tfmot.sparsity.keras.ConstantSparsity(
    #     target_sparsity=.7, begin_step=0)
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
        initial_sparsity=0, final_sparsity=.5, begin_step=0, end_step=5000, power=1)
    if not isinstance(layer, keras.layers.TimeDistributed):
        return tfmot.sparsity.keras.prune_low_magnitude(layer,pruning_schedule)
    return layer
#%% 
(X, X_train, X_test), (y, y_train, y_test), \
    (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(500/16*10**-6)

X_mini, y_mini = split_train_random(X_train, y_train, 20000, 200)

model = keras.models.load_model("./model_saves/pretrained_sequential")

smodel = make_LSTM_singular_model(model, hoyer=0.01, orthogonal=None, merged_kernel=False)
#%% regularization
smodel.compile(
    loss="mse",
    optimizer="adam"
)
smodel.fit(
    X_mini, y_mini,
    batch_size=32,
    validation_data=(X,y.reshape(1, -1, 1)),
    epochs=10
)
s = []
for layer in smodel.layers[:-1]: 
    s.append(layer.cell.kernel.numpy())
    s.append(layer.cell.recurrent_kernel.numpy())
#%% pruning
# prune_low_magnitude = tfmot.sparsity.keras.prune_low_magnitude

# smodel = keras.models.clone_model(smodel, clone_function=apply_pruning_to_LSTM)
# smodel.compile(
#     loss="mse",
#     optimizer="adam"
# )

# smodel.fit(X_mini, y_mini, batch_size=32, validation_data=(X_test,y_test), epochs=25, 
#             callbacks=[pruning_callbacks.UpdatePruningStep()])
#%% created reduced model
rmodel = make_LSTM_reduced_model(smodel, merged_kernel=False, cutoff=.05)

start_time = time.perf_counter()
fy = model.predict(X)
print("full model timing: " + str(time.perf_counter() - start_time) + " sec")
start_time = time.perf_counter()
ry = rmodel.predict(X)
print("reduced model timing: " + str(time.perf_counter() - start_time) + " sec")
# start_time = time.perf_counter()
# sy = smodel.predict(X)
# print("singular model timing: " + str(time.perf_counter() - start_time) + " sec")
#%% analysis\
from sklearn.metrics import mean_squared_error
from math import sqrt

# count weights in rmodel
reduced_weights = 0
for weight_matrix in rmodel.get_weights():
    reduced_weights += weight_matrix.size
full_weights = 0
for weight_matrix in model.get_weights():
    full_weights += weight_matrix.size

print("%d weights in full matrix"%full_weights)
print("%d weights in reduced matrix"%reduced_weights)
print("%f percent reduction in weights"%(100 - reduced_weights/full_weights*100))

ry_scaled = pin_scaler.inverse_transform(ry.reshape(-1, 1))
my_scaled = pin_scaler.inverse_transform(fy.reshape(-1, 1))
y_scaled = pin_scaler.inverse_transform(y.reshape(-1, 1))

plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t, y_scaled.flatten(), label = "reference",alpha=.8)
plt.plot(t, ry_scaled.flatten(), label = "reduced model")
plt.plot(t, my_scaled.flatten(), label = "full model")
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
# plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()

rmse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_scaled, ry_scaled)])/y_test.shape[0]
rrmse = sqrt(sum([mean_squared_error(y_t, y_p, squared=False)**2 for y_t, y_p in zip(y_scaled, ry_scaled)])/y_test.shape[0])
mmse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_scaled, my_scaled)])/y_test.shape[0]
mrmse = sqrt(sum([mean_squared_error(y_t, y_p, squared=False)**2 for y_t, y_p in zip(y_scaled, my_scaled)])/y_test.shape[0])

print("%f m RMSE of reduced model"%(rrmse))
print("%f m RMSE of full model"%(mrmse))
print("%f percent increase in RMSE"%(rrmse/mrmse*100-100))


rsnr = signaltonoise(y_scaled, ry_scaled)
msnr = signaltonoise(y_scaled, my_scaled)
nsnr = signaltonoise(my_scaled, ry_scaled, invert=True)

print("%f dB SNR of reduced model"%rsnr)
print("%f dB SNR of full model"%msnr)
print("%f dB reduction of model SNR"%(msnr-rsnr))
print("%f dB noise from full to reduced model"%(nsnr))

# start_time = time.perf_counter()
# sy = smodel.predict(X_test)
# print("singular model timing: " + str(time.perf_counter() - start_time) + " sec")
# plt.figure(figsize=(7,3.3))
# plt.title("LSTM prediction of pin location")
# plt.plot(t_test[0], sy[0], label = "predicted pin location")
# plt.plot(t_test[0], y_test[0], label = "actual pin location",alpha=.8)
# plt.xlabel("time [s]")
# plt.ylabel("pin location [m]")
# # plt.ylim((0.045, .23))
# plt.legend(loc=1)
# plt.tight_layout()