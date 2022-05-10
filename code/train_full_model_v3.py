import joblib
import pickle
import tensorflow.keras as keras
import matplotlib.pyplot as plt
# from math import isnan
import numpy as np
from numpy.random import randint
"""
Goals of this model:
    1. Train with an epoch-based method rather than stocastically.
    2. Input training data will be vector of last few acceleration datapoints.
This requires modifying the preprocess code, but I'll keep that in this file

Results show that stocastic training is much better than full-dataset epoch-based.
At least it converges much, much quicker. Inputting multiple training points
helps the model greatly, and quartered the RMSE.

"""
#%% preprocess
def preprocess():
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
    
    ds = 64 # downsampling factor
    
    # scaling data, which means that it must be unscaled to be useful
    from sklearn import preprocessing
    acc_scaler = sk.preprocessing.StandardScaler()
    acc_scaler.fit(acc.reshape(-1, 1))
    acc = acc_scaler.fit_transform(acc.reshape(-1, 1)).flatten()
    pin_scaler = sk.preprocessing.StandardScaler()
    pin_scaler.fit(pin.reshape(-1,1))
    pin_transform = pin_scaler.fit_transform(pin.reshape(-1,1)).flatten().astype(np.float32)
    
    y = np.array([pin_transform[(np.abs(pin_t - v)).argmin()] for v in acc_t])
    # remove data from before initial excitement (at 1.5 seconds)
    acc = acc[acc_t > 1.5]
    y = y[acc_t > 1.5]
    acc_t = acc_t[acc_t > 1.5]
    
    #reshape/downsample
    X = np.reshape(acc[:acc.size//ds*ds], (acc.size//ds, ds)).T
    acc_t_reshape = np.reshape(acc_t[:acc_t.size//ds*ds], (acc_t.size//ds, ds)).T
    y = np.reshape(y[:y.size//ds*ds], (y.size//ds, ds)).T
    return X, y, pin_scaler, acc_t_reshape
#%% load data
X, y, pin_scaler, t_test = preprocess()
X_rolled = np.expand_dims(X[0:64:4].T,axis=0)
y_rolled = np.expand_dims(y[0:64:4].T,axis=0)
for i in range(1,4):
    x_mini = np.expand_dims(X[i:64:4].T,axis=0)
    y_mini = np.expand_dims(y[i:64:4].T,axis=0)
    X_rolled = np.append(X_rolled, x_mini, axis=0)
    y_rolled = np.append(y_rolled, y_mini, axis=0)

X_train = X_rolled[1:,:,:]
X_test = np.expand_dims(X_rolled[1,:,:],0)
y_train = y_rolled[1:,:,:]
y_test = np.expand_dims(y_rolled[1,:,:],0)

#%% model is a splice of the period and amplitude for the lower two layers
def split_train_random(batch_size, train_len):
    runs = X_train.shape[0]
    run_size = X_train.shape[1]
    indices = [(randint(1, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
    X_mini = np.copy(np.array([X_train[index[0],index[1]:index[1]+train_len] for index in indices]))
    y_mini = np.copy(np.array([y_train[index[0],index[1]+train_len][0] for index in indices]))
    return X_mini, y_mini
print("making model...")
units = [30, 30, 15, 15]
model = keras.Sequential(
    [keras.layers.LSTM(units[0],return_sequences=True,input_shape=[None, 16])] + 
    [keras.layers.LSTM(i, return_sequences = True) for i in units[1:]] +
    [keras.layers.TimeDistributed(keras.layers.Dense(1))]
)


model.compile(loss="mse",
    optimizer="adam",
    metrics = ['accuracy']
)
print("training model...")

X_mini, y_mini = split_train_random(10000, 100)


model.fit(X_mini, y_mini, validation_data=(X_test,y_test), epochs=10)

# training on whole dataset at once:
# y_full = np.expand_dims(y_train[:,:,0],-1)
# model.fit(X_train, y_full, validation_data=(X_train,y_train), epochs=20)
#%%
# model = keras.models.load_model("./model_saves/pretrained_sequential")
#%% plots and evaluate accuracy
from sklearn.metrics import mean_squared_error
from math import sqrt
y_pred = model.predict(X_test)
y_pred_scaled = pin_scaler.inverse_transform(model.predict(X_test)[0]).T
y_test_scaled = pin_scaler.inverse_transform(y_test[:,:,0])

mse = mean_squared_error(y_test_scaled, y_pred_scaled, squared = True)
rmse = mean_squared_error(y_test_scaled, y_pred_scaled, squared = False)

mse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0]
rmse = sqrt(sum([mean_squared_error(y_t, y_p, squared=False)**2 for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0])

print("mean squared error: " + str(mse))
print("root mean squared error: " + str(rmse))

weight_sum = 0
for weights in model.trainable_weights:
    weight_sum += np.size(weights)

print("total weights: " + str(weight_sum)) 

plt.figure(figsize=(7,3.3))
plt.title("LSTM prediction of pin location")
plt.plot(t_test[0], y_pred_scaled[0], label = "predicted pin location")
plt.plot(t_test[0], y_test_scaled[0], label = "actual pin location",alpha=.8)
plt.xlabel("time [s]")
plt.ylabel("pin location [m]")
plt.ylim((0.045, .23))
plt.legend(loc=1)
plt.tight_layout()
plt.savefig("./plots/full_model_prediction.png", dpi=800)

model.save("./model_saves/")