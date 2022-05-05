w = wu[0]; u = wu[1]

w_left = tf.Variable(w[0], trainable=False, dtype='float32')
w_sigma = tf.Variable(w[1], trainable=False, dtype='float32')
w_right = tf.Variable(w[2], trainable=False, dtype='float32')
u_left = tf.Variable(u[0], trainable=False, dtype='float32')
u_sigma = tf.Variable(u[1], trainable=False, dtype='float32')
u_right = tf.Variable(u[2], trainable=False, dtype='float32')
b = tf.Variable(b, trainable=False, dtype='float32')

wr_i, wr_f, wr_c, wr_o = tf.split(w_right, num_or_size_splits=4, axis=1)
ws_i, ws_f, ws_c, ws_o = tf.split(
    w_sigma, num_or_size_splits=4)
wl_i, wl_f, wl_c, wl_o = tf.split(
    w_left, num_or_size_splits=4, axis=1)


x_i = backend.dot(inputs_i, wl_i)
x_f = backend.dot(inputs_f, wl_f)
x_c = backend.dot(inputs_c, wl_c)
x_o = backend.dot(inputs_o, wl_o)

x_i = x_i * ws_i
x_f = x_f * ws_f
x_c = x_c * ws_c
x_o = x_o * ws_o

x_i = backend.dot(x_i, wr_i)
x_f = backend.dot(x_f, wr_f)
x_c = backend.dot(x_c, wr_c)
x_o = backend.dot(x_o, wr_o)

b_i, b_f, b_c, b_o = tf.split(
    b, num_or_size_splits=4, axis=0)


x_i = backend.bias_add(x_i, b_i)
x_f = backend.bias_add(x_f, b_f)
x_c = backend.bias_add(x_c, b_c)
x_o = backend.bias_add(x_o, b_o)


h_tm1 = tf.Variable(np.ones((1,30)), dtype = 'float32')
c_tm1 = tf.Variable(np.ones((1,30)), dtype = 'float32')


h_tm1_i = h_tm1
h_tm1_f = h_tm1
h_tm1_c = h_tm1
h_tm1_o = h_tm1


ur_i, ur_f, ur_c, ur_o = tf.split(
    u_right, num_or_size_splits=4, axis=1)
us_i, us_f, us_c, us_o = tf.split(
    u_sigma, num_or_size_splits=4)
ul_i, ul_f, ul_c, ul_o = tf.split(
    u_left, num_or_size_splits=4, axis=1)

r_i = backend.dot(h_tm1_i, ul_i)
r_f = backend.dot(h_tm1_f, ul_f)
r_c = backend.dot(h_tm1_c, ul_c)
r_o = backend.dot(h_tm1_o, ul_o)

r_i = r_i * us_i
r_f = r_f * us_f
r_c = r_c * us_c
r_o = r_o * us_o

r_i = backend.dot(r_i, ur_i)
r_f = backend.dot(r_f, ur_f)
r_c = backend.dot(r_c, ur_c)
r_o = backend.dot(r_o, ur_o)

smodel = make_LSTM_singular_model(model)

def make_another_model(model):
    smodel = keras.models.Sequential()
    smodel.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for layer in model.layers:
        cell = LSTMCell(layer.units)
        lstm = SingularLSTM()
        lstm.cell = cell
        rnncell = keras.layers.RNN(cell)
        smodel.add(rnncell)
    dense_top = keras.layers.TimeDistributed(keras.layers.Dense(1))
    smodel.add(dense_top)
    
    
    
    
