import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import LSTMCell
import tensorflow.keras.backend as backend
import tensorflow.keras.backend as K
import numpy as np
from numpy import matmul
"""
v3. Now I want to get this to work by defining my own custom layer, following
this stackoverflow post:
https://stackoverflow.com/questions/54231440/define-custom-lstm-cell-in-keras
"""
# the class does not inherit from LSTM
class SingularLSTM(keras.layers.Layer):
    
    def __init__(self, units,  w=None,u=None, **kwargs):
        self.implementation = 0 # 0 for reduced, 1 for full (kernel)
        self.units = units
        self.state_size = units
        super(SingularLSTM, self).__init__(**kwargs)
        self.w_left = tf.Variable(w[0], trainable=False, dtype='float32')
        self.w_sigma = tf.Variable(w[1], trainable=False, dtype='float32')
        self.w_right = tf.Variable(w[2], trainable=False, dtype='float32')
        self.u_left = tf.Variable(u[0], trainable=False, dtype='float32')
        self.u_sigma = tf.Variable(u[1], trainable=False, dtype='float32')
        self.u_right = tf.Variable(u[2], trainable=False, dtype='float32')
        # self.h = tf.Variable(tf.zeros((kwargs.units), 'float32'),trainable
    
    # builds for the normal LSTM cell.
    def build(self, input_shape):
        self.kernel = self.add_weight(shape=(input_shape[-1], self.units),
                                      initializer='uniform',
                                      name='kernel')
        self.recurrent_kernel = self.add_weight(
            shape=(self.units, self.units),
            initializer='uniform',
            name='recurrent_kernel')
        self.built = True
    
    def call(self, inputs, states):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
        if(self.implementation == 1):
            if 0. < self.dropout < 1.:
                inputs = inputs * dp_mask[0]
        z = backend.dot(inputs, self.kernel)
        z += backend.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
            z = backend.bias_add(z, self.bias)
        
        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
        
        h = o * self.activation(c)
        return h, [h, c]

    