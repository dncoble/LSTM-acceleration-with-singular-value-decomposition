import keras.backend as K
import numpy as np
from numpy import matmul
"""
Classes and functions for SVD reduction. Please excuse my object-based obsession,
it's a side effect of my Java training.
"""
# use SVD to produce a matrix with a lower rank
def reduce_matrix_rank(a, rank):
    u, s, v = np.linalg.svd(a, full_matrices=True, compute_uv=True)
    s[range(rank, s.size)] = 0
    return (u * s) @ v

def reduce_two_step(a, rank):
    u, s, v = np.linalg.svd(a, full_matrices=True, compute_uv=True)
    s = np.diag(s)[:rank, :rank]
    v = v[:rank,:]
    u = u[:,:rank]
    b = u[:rank, :]
    c = u[rank:,:]
    return [b @ s @ v, c @ np.linalg.inv(b)]

# I want to implement my own LSTMCell object to make sure that I can control
# what accelerations tf.keras is using
class My_LSTM_Cell: 
    
    def __init__(self, units, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo):
        self.Wi = Wi; self.Wf = Wf; self.Wc = Wc; self.Wo = Wo # investigate transposes
        self.Ui = Ui; self.Uf = Uf; self.Uc = Uc; self.Uo = Uo
        self.bi = bi; self.bf = bf; self.bc = bc; self.bo = bo
        self.h = np.zeros((units, 1)) # make sure this is a column vector
        self.c = np.zeros((units, 1))
    
    # x is a numpy/tf column vector
    def single_step_forward_pass(self, x):
        i = K.sigmoid(self.Wi @ x + self.Ui @ self.h + self.bi)
        f = K.sigmoid(matmul(self.Wf, x) + matmul(self.Uf, self.h) + self.bf)
        g = K.tanh(matmul(self.Wc, x) + matmul(self.Uc, self.h) + self.bc)
        o = K.sigmoid(matmul(self.Wo, x) + matmul(self.Uo, self.h) + self.bo)
        self.c = f*self.c + i*g # element-wise multiplication
        self.h = o*K.tanh(self.c)
        return self.h

# implements My_LSTM_Cell so that I can use My_LSTM_Model for both
class Reduced_LSTM_Cell(My_LSTM_Cell):
    
    # matrices are lists with two elements of 2D nparrays that are the first
    # and second matrices of the two-step multiplication
    def __init__(self, units, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo):
        self.Wi = Wi; self.Wf = Wf; self.Wc = Wc; self.Wo = Wo
        self.Ui = Ui; self.Uf = Uf; self.Uc = Uc; self.Uo = Uo
        self.bi = bi; self.bf = bf; self.bc = bc; self.bo = bo
        self.n = units
        self.r = Ui[0].shape[0] # might be 1
        self.h = np.zeros((self.n, 1))
        self.c = np.zeros((self.n, 1))
        self.k = self.n - self.r
        # saving these as instance variables should speed up computation
        self.wix = np.empty((self.n,1))
        self.uix = np.empty((self.n,1))
        self.wfx = np.empty((self.n,1))
        self.ufx = np.empty((self.n,1))
        self.wcx = np.empty((self.n,1))
        self.ucx = np.empty((self.n,1))
        self.wox = np.empty((self.n,1))
        self.uox = np.empty((self.n,1))
        
    
    def single_step_forward_pass(self, x):
        self.wix[:self.r] = self.Wi[0] @ x
        self.wix[self.r:] =self.Wi[1] @ self.wix[:self.r]
        
        self.uix[:self.r] = self.Ui[0] @ self.h
        self.uix[self.r:] =self.Ui[1] @ self.uix[:self.r]
        
        self.wfx[:self.r] = self.Wf[0] @  x
        self.wfx[self.r:] =self.Wf[1] @ self.wfx[:self.r]
        
        self.ufx[:self.r] = self.Uf[0] @ self.h
        self.ufx[self.r:] = self.Uf[1] @ self.ufx[:self.r]
        
        self.wcx[:self.r] = self.Wc[0] @ x
        self.wcx[self.r:] = self.Wc[1] @ self.wcx[:self.r]
        
        self.ucx[:self.r] = self.Uc[0] @ self.h
        self.ucx[self.r:] = self.Uc[1] @ self.ucx[:self.r]
        
        self.wox[:self.r] = self.Wo[0] @ x
        self.wox[self.r:] = self.Wo[1] @ self.wox[:self.r]
        
        self.uox[:self.r] = self.Uo[0] @ self.h
        self.uox[self.r:] = self.Uo[1] @ self.uox[:self.r]
        
        i = K.sigmoid(self.wix + self.uix + self.bi)
        f = K.sigmoid(self.wfx + self.ufx + self.bf)
        g = K.tanh(self.wcx + self.ucx + self.bc)
        o = K.sigmoid(self.wox + self.uox + self.bo)
        
        self.c = f*self.c + i*g # element-wise multiplication
        self.h = o*K.tanh(self.c)
        return self.h
        

class My_LSTM_Model:
    
    def __init__(self, cells, dense_top):
        self.cells = cells
        self.dense_top = dense_top # this will be just the keras object
    
    def single_step_forward_pass(self, x):
        for cell in self.cells:
            x = cell.single_step_forward_pass(x)
        return self.dense_top.call(K.reshape(x, (1, 1, -1)))
    
    def multi_step_forward_pass(self, x):
        y = np.zeros(x.shape)
        for i in range(x.shape[0]):
            y[i, :] = self.single_step_forward_pass(x[i:i+1,:])
        return y

def make_My_LSTM_Cell(keras_layer):
    W, U, b = keras_layer.get_weights()
    units = U.shape[0]
    # I believe these must be transposed
    Wi = W[:,:units].T; Wf = W[:,units:units*2].T; Wc = W[:,units*2:units*3].T; Wo = W[:,units*3:].T
    Ui = U[:,:units].T; Uf = U[:,units:units*2].T; Uc = U[:,units*2:units*3].T; Uo = U[:,units*3:].T
    bi = np.expand_dims(b[:units],1); bf = np.expand_dims(b[units:units*2],1);
    bc = np.expand_dims(b[units*2:units*3],1); bo = np.expand_dims(b[units*3:],1)
    return My_LSTM_Cell(units, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo)

"""
My_LSTM_Cell and Reduced_LSTM_Cell have terrible performance, which can't be 
avoided. So instead I'm going to wrap a keras model with the methods that I 
want to modify it with. 
Doing it this way should make it easier to experiment with 'heuristics' for 
finding which sigmas to eliminate. Could even go crazy and make another ML
model for predicting them.
"""
class LSTM_wrapper():
    
    def __init__(self, model, scaler):
        self.scaler = scaler
        self.model = model
        self.model_ranks = np.full((4,2,4), 30) # change to model dimensions
        self.model_singular_values = get_model_singular_values(model)
    
    # fill in all the 
    """
    heuristic names:
        'absolute' order by value of singular values
    """
    def iterate_reduce_model(self, X, y, threshold=None,reductions=None,evaluate_every=1,\
                             heuristic='absolute'):
        from sklearn.metrics import mean_squared_error
        from math import sqrt
        
        check_threshold = (threshold != None)
        if(heuristic == 'absolute'):
            sorted_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(self.model_singular_values.ravel()), (4,2,4,30))))
        
        iterations =  sorted_indices.shape[0]- 20
        rmse = np.zeros(iterations//evaluate_every)
        weights_eliminated = np.zeros(iterations)
        running_weights = 0
        y_test_scaled = self.scaler.inverse_transform(y.squeeze())
        for i in range(reductions):
            if(i % evaluate_every == 0):
                y_pred_scaled = self.scaler.inverse_transform(self.model.predict(X).squeeze())
                mse = sum([mean_squared_error(y_t, y_p) for y_t, y_p in zip(y_test_scaled, y_pred_scaled)])/y.shape[0]
                rmse[i//evaluate_every] = sqrt(mse)
                weights_eliminated[i] = running_weights
                if(check_threshold and rmse[i//evaluate_every] > threshold):
                    break;
            rank = self.model_ranks[sorted_indices[i][0],sorted_indices[i][1],sorted_indices[i][2]] - 1
            self.model_ranks[sorted_indices[i][0],sorted_indices[i][1],sorted_indices[i][2]] = rank
            self.model = set_model_matrix_rank(self.model, sorted_indices[i][:-1], rank)
            running_weights += 2*30 - 2*rank - 1
            print(str(i) + " out of " + str(iterations) + " reductions performed.")
    
    def reduce_n_times(self, heuristic='absolute'):
        if(heuristic == 'absolute'):
            sorted_indices = np.squeeze(np.dstack(np.unravel_index(np.argsort(self.model_singular_values.ravel()), (4,2,4,30))))


# composed of LSTM layers and a dense top
def make_LSTM_Model(keras_model):
    cells = []
    for layer in keras_model.layers[:-1]:
        cells.append(make_My_LSTM_Cell(layer))
    dense_top = keras_model.layers[-1]
    return My_LSTM_Model(cells, dense_top)

def make_Reduced_LSTM_Cell(keras_layer, rank):
    W, U, b = keras_layer.get_weights()
    units = U.shape[0]
    Wi = reduce_two_step(W[:,:units].T, rank)
    Wf = reduce_two_step(W[:,units:units*2].T, rank)
    Wc = reduce_two_step(W[:,units*2:units*3].T, rank)
    Wo = reduce_two_step(W[:,units*3:].T, rank)
    Ui = reduce_two_step(U[:,:units].T, rank)
    Uf = reduce_two_step(U[:,units:units*2].T, rank)
    Uc = reduce_two_step(U[:,units*2:units*3].T, rank)
    Uo = reduce_two_step(U[:,units*3:].T, rank)
    bi = np.expand_dims(b[:units],1); bf = np.expand_dims(b[units:units*2],1);
    bc = np.expand_dims(b[units*2:units*3],1); bo = np.expand_dims(b[units*3:],1)
    return Reduced_LSTM_Cell(units, Wi, Wf, Wc, Wo, Ui, Uf, Uc, Uo, bi, bf, bc, bo)


# index will be a tuple (cell, W/U, gate)
def set_model_matrix_rank(model, index, rank):
    W, U, b = model.layers[index[0]].get_weights()
    units = U.shape[0]
    
    m = [W,U][index[1]][:,units*index[2]:units*(index[2]+1)] # no transpose
    [W,U][index[1]][:,units*index[2]:units*(index[2]+1)] = reduce_matrix_rank(m, rank)
    model.layers[index[0]].set_weights((W,U,b))
    return model
    

def get_model_singular_values(model):
    # a bunch of for loops
    # assumes a square model
    layers = len(model.layers) - 1
    units = model.layers[0].get_weights()[1].shape[0]
    rtrn = np.zeros((layers, 2, 4, units))
    for i in range(layers):
        W, U, b = model.layers[i].get_weights()
        for j in range(2):
            m = [W,U][j]
            for k in range(4):
                rtrn[i,j,k,:] = np.linalg.svd(m, full_matrices=True,compute_uv=False)
    return rtrn
    

# testing out np.linalg.svd
# if __name__ == '__main__':
    # A = np.array([[1, 3, 5], [2, 2, 2], [5, 8, 9]])
    # print("A: ")
    # print(A)
    # print("rank: " + str(np.linalg.matrix_rank(A)))
    # U, S, V = np.linalg.svd(A, full_matrices=True, compute_uv=True)
    # # print(U); print(S); print(V)
    # drop_point = 2
    # zero_indices = range(drop_point, S.size)
    # S[zero_indices] = 0
    # A_reconstruct = (U * S) @ V
    # print("reconstructed A: ")
    # print(A_reconstruct)
    # print("reconstructed rank: " + str(np.linalg.matrix_rank(A_reconstruct)))
    