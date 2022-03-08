import tensorflow as tf
import tensorflow.keras as keras
from keras.layers.rnn.lstm import LSTMCell
import keras.backend as backend
"""
after the full model is trained, we 
this technically counts as another training step because data will be passed
over the model again to determine which weights should be dropped
"""
class SingularLSTMCell(LSTMCell):
    
    def __init__(self, **kwargs):
        super(SingularLSTMCell, self).__init__(**kwargs)
        self.w_left = tf.Variable(kwargs.w_left, 'float32',trainable=False)
        self.w_sigma = tf.Variable((kwargs.w_sigma))
        self.w_right = tf.Variable(kwargs.w_right, 'float32',trainable=False)
        self.u_left = tf.Variable(kwargs.u_left, 'float32',trainable=False)
        self.u_right = tf.Variable(kwargs.u_right, 'float32',trainable=False)
        self.h = tf.Variable(tf.zeros((kwargs.units), 'float32'))
        self.c = tf.Variable(tf.zeros((kwargs.units), 'float32'))
    
    """ taken and modified from keras source code"""
    def call(self, inputs, states, training=None):
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state
    
        dp_mask = self.get_dropout_mask_for_cell(inputs, training, count=4)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            h_tm1, training, count=4)
        
        if 0. < self.dropout < 1.:
          inputs = inputs * dp_mask[0]
        
        z = backend.dot(inputs, self.w_right)
        z = backend.dot(z, )
        
        z += backend.dot(h_tm1, self.recurrent_kernel)
        if self.use_bias:
          z = backend.bias_add(z, self.bias)
  
        z = tf.split(z, num_or_size_splits=4, axis=1)
        c, o = self._compute_carry_and_output_fused(z, c_tm1)
    
        h = o * self.activation(c)
        return h, [h, c]