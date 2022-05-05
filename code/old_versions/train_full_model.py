import json
import pickle
import joblib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.keras as keras
import sklearn as sk
""" In this file I create a full LSTM model (initially of size 30-30-30-30), 
and train it. I tried to remove the pretrained toy models as a step, but it
didn't seem to work, so they're staying. But moving all this to a new file
lets me expand the SVD stuff in svd_accelerate.py without overcrowding that
script.
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
#%% toy models - period and amplitude
print("loading toy models...")
amplitude_model = keras.models.load_model('./model_saves/amplitude')
frequency_model = keras.models.load_model('./model_saves/frequency')
#%% model is a splice of the period and amplitude for the lower two layers
print("making model...")
f_units = 20
a_units = 10
m_units = 20
n_units = 10
x_input = keras.Input(shape = [None, 1], name = 'x')

f = keras.Sequential([
    keras.layers.LSTM(f_units, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(f_units, return_sequences = True)],
    name='f')(x_input)
a = keras.Sequential([
    keras.layers.LSTM(a_units, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(a_units, return_sequences = True)],
    name='a')(x_input)
m = keras.layers.LSTM(m_units, name='m', return_sequences=True)(f)
n = keras.layers.LSTM(n_units, name='n', return_sequences=True)(a)
concat = keras.layers.Concatenate(axis=2, name='concat')([m,n])
d = keras.layers.TimeDistributed(keras.layers.Dense(1), name='d')(concat)

model = keras.Model(
    inputs=[x_input],
    outputs=[d]
)
model.get_layer('f').layers[0].set_weights(frequency_model.layers[0].get_weights())
model.get_layer('f').layers[1].set_weights(frequency_model.layers[1].get_weights())
model.get_layer('a').layers[0].set_weights(amplitude_model.layers[0].get_weights())
model.get_layer('a').layers[1].set_weights(amplitude_model.layers[1].get_weights())
#%% train model
frame = 0 # i want to make a gif of training
def plot_evaluate(X_test, y_test, t_test, model, plot_fig = True, savfig = False, savpath = "plot.png"):
    from sklearn.metrics import mean_squared_error
    y_pred_scaled = pin_scaler.inverse_transform(model.predict(X_test).squeeze())
    y_test_scaled = pin_scaler.inverse_transform(y_test.squeeze())
    
    mse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0]
    rmse = sum([mean_squared_error(y_t, y_p, squared=False) for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y_test.shape[0]
    
    plt.figure(figsize=(7,3.3))
    plt.title("LSTM prediction of pin location")
    plt.plot(t_test[0], y_pred_scaled[0], label = "predicted pin location")
    plt.plot(t_test[0], y_test_scaled[0], label = "actual pin location",alpha=.8)
    plt.xlabel("time [s]")
    plt.ylabel("pin location [m]")
    plt.ylim((0.045, .23))
    plt.legend(loc=1)
    plt.tight_layout()
    if(savfig):
        plt.savefig(savpath, dpi=800)
    if(not plot_fig):
        plt.close()
    print("mean squared error: " + str(mse))
    print("root mean squared error: " + str(rmse))
    return rmse, mse

def fit(X_train, y_train, t_train, model, X_test=None, y_test=None, t_test=None,
        filepath = "./model_saves/model", learning_rate=0.01, train_len=100, 
        batch_size=32, total_batches=500,
        evaluate_every=None):
    global frame
    print("training model...")
    model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
            filepath = filepath,
            save_weights_only = True,
            monitor = "loss",
            mode = "min",
            save_freq = 1,
            save_best_only = True)
    model.save_weights(filepath)
    optimizer = keras.optimizers.SGD(clipvalue = .5, learning_rate=learning_rate, momentum=.9)
    model.compile(loss = "mean_squared_error",
                  optimizer = optimizer)
    from math import isnan
    from numpy.random import randint
    
    runs = X_train.shape[0] # i'm treating each split of the data as a different 'run' of the experiment
    run_size = X_train.shape[1]
    best_rmse = None
    for i in range(total_batches):
        if(i % 100 == 0):
            print("{} mini-batches completed".format(i))
        indices = [(randint(1, runs), randint(0, run_size - train_len)) for i in range(batch_size)]
        X_mini = np.array([X_train[index[0],index[1]:index[1]+train_len] for index in indices])
        y_mini = np.array([y_train[index[0],index[1]+train_len] for index in indices])
        hist = model.fit(X_mini, y_mini, epochs=1, batch_size=batch_size,\
                         callbacks = [model_checkpoint_callback], verbose = 0)
        if(isnan(hist.history['loss'][0])):
            print("training failure")
            # broken_on_error = True; break
            model.load_weights(filepath)
        if(evaluate_every != None and i % evaluate_every == 0 and i != 0):
            rmse = plot_evaluate(X_test, y_test, t_test, model, savfig=True, savpath="./plots/plt" + str(frame))[0]
            frame += 1
            if(best_rmse == None or rmse < best_rmse):
                model.save_weights("./model_saves/pin_loc" + str(i))
                best_rmse = rmse
            
    return model

model.get_layer('f').trainable = True
model.get_layer('a').trainable = True
model.get_layer('m').trainable = True
model.get_layer('n').trainable = True
model = fit(X_train, y_train, t_train, model, X_test=X_test, y_test=y_test, t_test=t_test,
            total_batches = 1800, evaluate_every=200)
plot_evaluate(X_test, y_test, t_test, model, savfig=True, savpath="./plots/plt" + str(frame)); frame +=1;
print("training complete.")
model.save("./model_saves/pretrained_split")
#%% turn LSTM sequential
from sequentialize import merge_parallel_cell_weights
# so this was back when i had t instead of f
print("making sequential model")
t_units = f_units
seq_model = keras.models.Sequential([
    keras.layers.LSTM(30, return_sequences = True, input_shape = [None, 1]),
    keras.layers.LSTM(30, return_sequences = True),
    keras.layers.LSTM(30, return_sequences = True),
    keras.layers.LSTM(30, return_sequences = True),
    keras.layers.TimeDistributed(keras.layers.Dense(1))
])

weights_1 = merge_parallel_cell_weights(
    model.get_layer('f').layers[0].get_weights(),
    model.get_layer('a').layers[0].get_weights(),
    same_x = True)
weights_2 = merge_parallel_cell_weights(
    model.get_layer('f').layers[1].get_weights(),
    model.get_layer('a').layers[1].get_weights(),
    same_x = False)
weights_3 = merge_parallel_cell_weights(
    model.get_layer('m').get_weights(),
    model.get_layer('n').get_weights(),
    same_x = False)

seq_model.layers[0].set_weights(weights_1)
seq_model.layers[1].set_weights(weights_2)
seq_model.layers[2].set_weights(weights_3)
#%% train with sequential model
model = seq_model
plot_evaluate(X_test, y_test, t_test, model, plot_fig=True, savfig=False)
model = fit(X_train, y_train, t_train, model, X_test=X_test, y_test=y_test,\
            t_test=t_test, total_batches = 1000, evaluate_every=200)
# plot_evaluate(savfig=True, savpath="./plots/" + str(frame)); frame +=1;
model.save("./model_saves/pretrained_sequential")