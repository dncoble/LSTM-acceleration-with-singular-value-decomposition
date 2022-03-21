import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTMCell
import tensorflow.keras.backend as backend
import tensorflow.keras.backend as K
import numpy as np
from numpy import matmul
"""
after the full model is trained, we do a reducing step where the
singular values are pruned.
this technically counts as another training step because data will be passed
over the model again to determine which weights should be dropped
these classes do not work with GPU
"""
class SingularLSTMCell(LSTMCell):
    
    def __init__(self, units, w=None,u=None, b=None, **kwargs):
        super(SingularLSTMCell, self).__init__(units, **kwargs)
        self.w_left = tf.Variable(w[0], trainable=False, dtype='float32')
        self.w_sigma = tf.Variable(w[1], trainable=False, dtype='float32')
        self.w_right = tf.Variable(w[2], trainable=False, dtype='float32')
        self.u_left = tf.Variable(u[0], trainable=False, dtype='float32')
        self.u_sigma = tf.Variable(u[1], trainable=False, dtype='float32')
        self.u_right = tf.Variable(u[2], trainable=False, dtype='float32')
        self.b = tf.Variable(b, trainable=False, dtype='float32')
        # self.h = tf.Variable(tf.zeros((kwargs.units), 'float32'),trainable=False)
        # self.c = tf.Variable(tf.zeros((kwargs.units), 'float32'),trainable=False)
    
    """ taken and modified from keras source code"""
    def call(self, inputs, states, training=None):
        print("reaching here classically")
        tf.print("correct call function")
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
    
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)
        
        if 0 < self.dropout < 1.:
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i = inputs
            inputs_f = inputs
            inputs_c = inputs
            inputs_o = inputs
        
        wr_i, wr_f, wr_c, wr_o = tf.split(
            self.w_right, num_or_size_splits=4, axis=1)
        ws_i, ws_f, ws_c, ws_o = tf.split(
            self.w_sigma, num_or_size_splits=4)
        wl_i, wl_f, wl_c, wl_o = tf.split(
            self.w_left, num_or_size_splits=4, axis=1)
        
        x_i = backend.dot(wr_i, inputs_i)
        x_f = backend.dot(wr_f, inputs_f)
        x_c = backend.dot(wr_c, inputs_c)
        x_o = backend.dot(wr_o, inputs_o)
        
        x_i = x_i * ws_i
        x_f = x_f * ws_f
        x_c = x_c * ws_c
        x_o = x_o * ws_o
        
        x_i = backend.dot(x_i, wl_i)
        x_f = backend.dot(x_f, wl_f)
        x_c = backend.dot(x_c, wl_c)
        x_o = backend.dot(x_o, wl_o)
        
        if self.use_bias:
            b_i, b_f, b_c, b_o = tf.split(
                self.b, num_or_size_splits=4, axis=0)
            x_i = x_i + b_i
            x_f = x_f + b_f
            x_c = x_c + b_c
            x_o = x_o + b_o
  
        if 0 < self.recurrent_dropout < 1.:
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i = h_tm1
            h_tm1_f = h_tm1
            h_tm1_c = h_tm1
            h_tm1_o = h_tm1
        
        ur_i, ur_f, ur_c, ur_o = tf.split(
            self.u_right, num_or_size_splits=4, axis=1)
        us_i, us_f, us_c, us_o = tf.split(
            self.u_sigma, num_or_size_splits=4)
        ul_i, ul_f, ul_c, ul_o = tf.split(
            self.u_left, num_or_size_splits=4, axis=1)
        
        r_i = backend.dot(h_tm1_i, ur_i)
        r_f = backend.dot(h_tm1_f, ur_f)
        r_c = backend.dot(h_tm1_c, ur_c)
        r_o = backend.dot(h_tm1_o, ur_o)
        
        r_i = r_i * us_i
        r_f = r_f * us_f
        r_c = r_c * us_c
        r_o = r_o * us_o
        
        r_i = backend.dot(r_i, ul_i)
        r_f = backend.dot(r_f, ul_f)
        r_c = backend.dot(r_c, ul_c)
        r_o = backend.dot(r_o, ul_o)
        
        i = self.recurrent_activation(x_i + r_i)
        f = self.recurrent_activation(x_f + r_f)
        c = f * c_tm1 + i * self.activation(x_c + r_c)
        o = self.recurrent_activation(x_o + r_o)
        
        # if 0. < self.dropout < 1.:
        #   inputs = inputs * dp_mask[0]
        
        # z = backend.dot(inputs, self.w_right)
        # z = backend.multiply(z, self.w_sigma)
        # z = backend.dot(z, self.w_left)
        
        # k = backend.dot(h_tm1, self.u_right)
        # k = backend.multiply(k, self.u_sigma)
        # k = backend.dot(k, self.u_left)
        
        # z += k
        # if self.use_bias:
        #   z = backend.bias_add(z, self.bias)
  
        # z = tf.split(z, num_or_size_splits=4, axis=1)
        # c, o = self._compute_carry_and_output_fused(z, c_tm1)
    
        h = o * self.activation(c)
        
        return h, [h, c]

"""
goes into the model and changes the LSTMCell objects to SingularLSTMCell.
assumes all layers in the model are LSTMs except the last
DON'T USE THIS FUNCTION, use instead make_LSTM_singular_model
"""
def convert_LSTM_to_singular(model):
    for layer in model.layers[:-1]:
        # print('count')
        w, u, b = layer.get_weights()
        units = u.shape[0]
        w_split = [w[:,:units],w[:,units:units*2],w[:,units*2:units*3],w[:,units*3:]]
        u_split = [u[:,:units],u[:,units:units*2],u[:,units*2:units*3],u[:,units*3:]]
        
        wu = []
        for split in [w_split, u_split]:
            lefts = []
            sigmas = []
            rights = []
            for mat in split:
                left, sigma, right = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
                lefts.append(left.T)
                sigmas.append(sigma)
                rights.append(right.T)
            unsplit_left = lefts[0]
            unsplit_sigma = sigmas[0]
            unsplit_right = rights[0]
            for left, sigma, right in zip(lefts[1:], sigmas[1:], rights[1:]):
                unsplit_left = np.append(unsplit_left, left, axis=1) # maybe wrong axis?
                unsplit_sigma = np.append(unsplit_sigma, sigma)
                unsplit_right = np.append(unsplit_right, right,axis=1)
            wu.append([unsplit_left, unsplit_sigma, unsplit_right])
        
        singular_cell = SingularLSTMCell(units, w=wu[0],u=wu[1])
        singular_cell.kernel = tf.Variable(w)
        singular_cell.recurrent_kernel = tf.Variable(u)
        singular_cell.bias = tf.Variable(b)
        
        layer.cell = singular_cell
    return model

def make_LSTM_singular_model(model):
    smodel = keras.models.Sequential()
    smodel.add(keras.layers.InputLayer(input_shape=[None,1]))
    for layer in model.layers[:-1]:
        print('count')
        w, u, b = layer.get_weights()
        units = u.shape[0]
        w_split = [w[:,:units],w[:,units:units*2],w[:,units*2:units*3],w[:,units*3:]]
        u_split = [u[:,:units],u[:,units:units*2],u[:,units*2:units*3],u[:,units*3:]]
        
        wu = []
        for split in [w_split, u_split]:
            lefts = []
            sigmas = []
            rights = []
            for mat in split:
                left, sigma, right = np.linalg.svd(mat, full_matrices=True, compute_uv=True)
                lefts.append(left)
                sigmas.append(sigma)
                rights.append(right)
            unsplit_left = lefts[0]
            unsplit_sigma = sigmas[0]
            unsplit_right = rights[0]
            for left, sigma, right in zip(lefts[1:], sigmas[1:], rights[1:]):
                unsplit_left = np.append(unsplit_left, left, axis=1) # maybe wrong axis?
                unsplit_sigma = np.append(unsplit_sigma, sigma)
                unsplit_right = np.append(unsplit_right, right,axis=1)
            wu.append([unsplit_left, unsplit_sigma, unsplit_right])
        
        # b = np.expand_dims(b, axis=1)
        cell = SingularLSTMCell(units, w=wu[0],u=wu[1],b=b)
        smodel.add(keras.layers.RNN(cell))
    
    dense_top = keras.layers.TimeDistributed(keras.layers.Dense(1))
    smodel.add(dense_top)
    return smodel