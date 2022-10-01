import joblib
import pickle
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import matplotlib as mpl
# from math import isnan
import numpy as np
from numpy.random import randint
import math
import scipy
from scipy import signal
from sklearn.metrics import mean_squared_error
"""
The LSTM is trained on data sampled at 500 us. Inputs will
be 16-element vectors sampled at 500/16 us. We'll use a 40-40-40-40 unit 
LSTM and training elements are .1s = 200 samples long. The model is trained
on the square and sinusuoid profiles and the impulses is left for validation.

This training scheme worked well for my LSTM matrix training.

now using TensorFlow 2.10.0
"""
#%% preprocess
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
def signaltonoise(signal, noisy_signal, dB=True):
    noise = signal - noisy_signal
    a_sig = math.sqrt(np.mean(np.square(signal)))
    a_noise = math.sqrt(np.mean(np.square(noise)))
    snr = (a_sig/a_noise)**2
    if(not dB):
        return snr
    return 10*math.log(snr, 10)
#%% train model
(X, X_train, X_test), (y, y_train, y_test), \
    (t, t_test, t_train), pin_scaler, acc_scaler = preprocess(500/16*10**-6)

units_structure = [40, 40, 40, 40];

# model = keras.Sequential(
#     [keras.layers.LSTM(units_structure[0],return_sequences=True,input_shape=[None, 16])] + 
#     [keras.layers.LSTM(i, return_sequences = True) for i in units_structure[1:]] +
#     [keras.layers.TimeDistributed(keras.layers.Dense(1))]
# )
model = keras.Sequential([
    keras.layers.LSTM(units_structure[0],return_sequences=True,input_shape=[None, 16]),
    keras.layers.LSTM(units_structure[1],return_sequences=True),
    keras.layers.LSTM(units_structure[2],return_sequences=True),
    keras.layers.LSTM(units_structure[3],return_sequences=False),
    keras.layers.Dense(1)
])
model.compile(loss="mse",
    optimizer="adam",
)

X_mini, y_mini = split_train_random(X_train, y_train, 20000, 200)

model.fit(
    X_mini, y_mini,
    epochs=30,
    # validation_data=(X, y.reshape(1, -1, 1)),
)

model.save("./model_saves/pretrained_sequential")


#%% analysis
model = keras.models.load_model("./model_saves/pretrained_sequential")
model2 = keras.Sequential([
    keras.layers.LSTM(units_structure[0],return_sequences=True,trainable=False,input_shape=[None, 16]),
    keras.layers.LSTM(units_structure[1],return_sequences=True,trainable=False),
    keras.layers.LSTM(units_structure[2],return_sequences=True,trainable=False),
    keras.layers.LSTM(units_structure[3],return_sequences=True,trainable=False),
    keras.layers.Dense(1)
])
for i, layer in enumerate(model.layers):
    model2.layers[i].set_weights(layer.get_weights())

model = model2
pred = pin_scaler.inverse_transform(model.predict(X)[0])
true = pin_scaler.inverse_transform(np.expand_dims(y,-1))

snr = signaltonoise(true, pred)
rmse = mean_squared_error(true, pred, squared=False)
nrmse = rmse/(np.max(true) - np.min(true))

print("SNR: " + str(snr))
print("RMSE: " + str(rmse))
print("NRMSE: " + str(nrmse))

plt.figure(figsize=(6, 3))
plt.plot(t, true, label='reference')
plt.plot(t, pred, '--', label='full model prediction')
plt.xlabel("time (s)")
plt.ylabel("displacement (m)")
plt.tight_layout()
plt.legend()