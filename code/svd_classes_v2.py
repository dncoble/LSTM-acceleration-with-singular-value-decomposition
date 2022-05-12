import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTMCell
import tensorflow.keras.backend as backend
# from keras.layers.rnn import gru_lstm_utils
import numpy as np
from numpy import matmul
"""
after the full model is trained, we do a reducing step where the
singular values are pruned.
this technically counts as another training step because data will be passed
over the model again to determine which weights should be dropped

SingleLSTMCell and SingleLSTM are modified from the keras source files, and
probably aren't fully 'stable', but for my uses they should be fine. The 
call() function for SingleLSTMCell does not work with GPUs, and SingleLSTM
shouldn't allow you to use GPU. 

TensorFlow 2.5.0
"""
class SingularLSTMCell(LSTMCell):
    
    def __init__(self, units, w=None,u=None, b=None, **kwargs):
        super(SingularLSTMCell, self).__init__(units, **kwargs)
        self.w_left = tf.Variable(w[0], trainable=False, dtype='float32')
        self.w_sigma = tf.Variable(w[1], trainable=True, dtype='float32')
        self.w_right = tf.Variable(w[2], trainable=False, dtype='float32')
        self.u_left = tf.Variable(u[0], trainable=False, dtype='float32')
        self.u_sigma = tf.Variable(u[1], trainable=True, dtype='float32')
        self.u_right = tf.Variable(u[2], trainable=False, dtype='float32')
        self.b = tf.Variable(b, trainable=False, dtype='float32')
        # self.h = tf.Variable(tf.zeros((kwargs.units), 'float32'),trainable=False)
        # self.c = tf.Variable(tf.zeros((kwargs.units), 'float32'),trainable=False)
    
    """ modified from keras source code """
    def call(self, inputs, states, training=None):
        # tf.print("correct call function")
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
        
        
        # keras uses left multiplication of horizontal (row) vectors
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
        
        if self.use_bias:
            b_i, b_f, b_c, b_o = tf.split(
                self.b, num_or_size_splits=4, axis=0)
            x_i = backend.bias_add(x_i, b_i)
            x_f = backend.bias_add(x_f, b_f)
            x_c = backend.bias_add(x_c, b_c)
            x_o = backend.bias_add(x_o, b_o)
  
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

class SingularLSTM(keras.layers.LSTM):
    
    def __init__(self, units, cell=None, **kwargs):
        super(SingularLSTM, self).__init__(units, **kwargs)
        self.cell = cell
    
    """ copied from the non-gpu portion of keras.layers.LSTM call() function"""
    def call(self, inputs, mask=None, training=None, initial_state=None):
        # The input should be dense, padded with zeros. If a ragged input is fed
        # into the layer, it is padded and the row lengths are used for masking.
        # inputs, row_lengths = backend.convert_inputs_if_ragged(inputs)
        # is_ragged_input = (row_lengths is not None)
        # self._validate_args_if_ragged(is_ragged_input, mask)
        
        # LSTM does not support constants. Ignore it during process.
        inputs, initial_state, _ = self._process_inputs(inputs, initial_state, None)
        
        if isinstance(mask, list):
          mask = mask[0]
        
        input_shape = backend.int_shape(inputs)
        timesteps = input_shape[0] if self.time_major else input_shape[1]
        
        # Fall back to use the normal LSTM.
        kwargs = {'training': training}
        self._maybe_reset_cell_dropout_mask(self.cell)
        
        def step(inputs, states):
          return self.cell(inputs, states, **kwargs)
        
        last_output, outputs, states = backend.rnn(
            step,
            inputs,
            initial_state,
            constants=None,
            go_backwards=self.go_backwards,
            mask=mask,
            unroll=self.unroll,
            input_length=timesteps,
            time_major=self.time_major,
            zero_output_for_mask=self.zero_output_for_mask)
        # runtime = gru_lstm_utils.runtime(gru_lstm_utils.RUNTIME_UNKNOWN)
        if self.stateful:
            updates = [
                  tf.compat.v1.assign(self_state, tf.cast(state, self_state.dtype))
                  for self_state, state in zip(self.states, states)
            ]
            self.add_update(updates)
        
        if self.return_sequences:
            output = outputs
        else:    
            output = last_output
        
        if self.return_state:
          return [output] + list(states)
        # elif self.return_runtime: # just gonna comment this out
        #   return output, runtime
        return output
    
    def get_prunable_weights(self):
        return [self.cell.w_sigma, self.cell.u_sigma]

class PrunableTimeDistributed(keras.layers.TimeDistributed):
    
    def __init__(self, layer, **kwargs):
        super(PrunableTimeDistributed, self).__init__(layer, **kwargs)
        self.layer = layer
    
    def get_prunable_weights(self):
        return self.layer.layer.weights
    
    

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
    smodel.add(keras.layers.InputLayer(input_shape=[None, 1]))
    for layer in model.layers[:-1]:
        w, u, b = layer.get_weights()
        
        # w = np.expand_dims(w, -1)
        # u = np.expand_dims(u, -1)
        
        # units = u.shape[0]
        units = layer.units
        
        w_split = [w[:,:units],w[:,units:units*2],w[:,units*2:units*3],w[:,units*3:]]
        u_split = [u[:,:units],u[:,units:units*2],u[:,units*2:units*3],u[:,units*3:]]
        
        wu = []
        for split in [w_split, u_split]:
            lefts = []
            sigmas = []
            rights = []
            for mat in split:
                left, sigma, right = np.linalg.svd(mat, full_matrices=False, compute_uv=True)
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
        
        # b = np.expand_dims(b, axis=-1).T
        # wu[0] = np.expand_dims(wu[0], 0)
        # wu[1] = np.expand_dims(wu[1], 0)
        
        cell = SingularLSTMCell(units, w=wu[0],u=wu[1],b=b)
        lstm = SingularLSTM(units, cell=cell, return_sequences=True)
        smodel.add(lstm)
    
    dense_top = keras.layers.TimeDistributed(keras.layers.Dense(1))
    smodel.add(dense_top)
    dense_top.set_weights([
            model.layers[-1].weights[0].numpy(),
            model.layers[-1].weights[1].numpy()
        ])
    return smodel