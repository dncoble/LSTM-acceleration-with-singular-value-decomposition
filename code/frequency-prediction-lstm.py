#%%
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#%%
"""
while i can't get DROPBEAR to work, i'm trying this more simple projects:
an LSTM to predict the period (inverse of frequency) from a generated
time series.

let delta t be 1 between each point
period will be between 2 and 10
phase varies across 0-2pi, amplitude 0-2
"""
""" y_type can be one of 'period', 'amplitude', 'frequency'"""
def generate_time_series(batch_size, n_steps, y_type = 'period'):
    T = np.random.rand(1, batch_size, 1) * 8 + 2
    phase = np.random.rand(1, batch_size, 1)*2*np.pi
    A = np.random.rand(1, batch_size, 1)*9.8 + .2
    time = np.linspace(0, n_steps, n_steps)
    series = A * np.sin((time - phase)*2*np.pi/T)
    series += 0.1 * (np.random.rand(1, batch_size, n_steps) - .5)
    rtrn = np.expand_dims(np.squeeze(series.astype(np.float32)), axis=2)
    if(y_type == 'amplitude'):
        return rtrn, A.flatten()
    if(y_type == 'frequency'):
        return rtrn, 1/T.flatten()
    return rtrn, T.flatten()

np.random.seed(42)

y_type = 'frequency'
n_steps = 75
X, y = generate_time_series(10000, n_steps + 1, y_type=y_type)
X_train = X[:7000]; y_train = y[:7000]
X_test = X[7000:]; y_test = y[7000:]
#%% model
model = keras.models.Sequential([
    keras.layers.LSTM(20, name="cell_1", return_sequences=True, input_shape=[None, 1]),
    keras.layers.LSTM(20, name="cell_2", return_sequences=True),
    keras.layers.Dense(1)
])

def last_time_step_mse(y_true, y_pred):
    return keras.metrics.mean_squared_error(y_true[:, -1], y_pred[:, -1])

model.compile(loss="mse", optimizer="adam")
history = model.fit(X_train, y_train, epochs = 20,
                    validation_data=(X_test, y_test))

#%%
y_pred = model.predict(X_test)[:,-1].flatten()

plt.figure(figsize=(4,3.4))
plt.scatter(y_test, y_pred, s=2, label = 'Test Data')
plt.plot(np.linspace(0, 25, 50), np.linspace(0, 25, 50), label = 'y=x', c='k')
plt.xlim(0.0, 0.5)
plt.ylim(0.0, 0.5)
if(y_type == 'amplitude'):
    plt.xlim(0.0, 10)
    plt.ylim(0.0, 10)
plt.legend()
plt.title("LSTM model predicts " + y_type)
plt.ylabel("true " + y_type)
plt.xlabel("predicted " + y_type)
plt.tight_layout()
plt.savefig("predicting_" + y_type + ".png", dpi=500)
#%% save model
model.save(y_type)