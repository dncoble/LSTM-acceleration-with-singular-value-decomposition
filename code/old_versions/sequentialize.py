# completely untested right now.
def merge_parallel_cell_weights(weights_1, weights_2, same_x=False):
    import numpy as np
    w1, u1, b1 = weights_1
    w2, u2, b2 = weights_2
    units_1 = u1.shape[0]
    units_2 = u2.shape[0]
    xdim_1 = w1.shape[0]
    xdim_2 = w2.shape[0]
    
    if(same_x):
        sw = np.append(w1[:,:units_1],w2[:,:units_2], axis=1)
        sw = np.append(sw, w1[:,units_1:units_1*2], axis=1)
        sw = np.append(sw, w2[:,units_2:units_2*2], axis=1)
        sw = np.append(sw, w1[:,units_1*2:units_1*3], axis=1)
        sw = np.append(sw, w2[:,units_2*2:units_2*3], axis=1)
        sw = np.append(sw, w1[:,units_1*3:], axis=1)
        sw = np.append(sw, w2[:,units_2*3:], axis=1)
    else:
        sw = np.insert(w1, units_1, np.zeros((units_2, xdim_1)), axis=1)
        sw = np.insert(sw, 2*units_1+units_2, np.zeros((units_2, xdim_1)), axis=1)
        sw = np.insert(sw, 3*units_1+2*units_2, np.zeros((units_2, xdim_1)), axis=1)
        sw = np.insert(sw, 4*units_1+3*units_2, np.zeros((units_2, xdim_1)), axis=1)
        sw_ = np.insert(w2, 0, np.zeros((units_1, xdim_2)), axis=1)
        sw_ = np.insert(sw_, units_1+units_2, np.zeros((units_1, xdim_2)), axis=1)
        sw_ = np.insert(sw_, 2*units_1+2*units_2, np.zeros((units_1, xdim_2)), axis=1)
        sw_ = np.insert(sw_, 3*units_1+3*units_2, np.zeros((units_1, xdim_2)), axis=1)
        sw = np.append(sw, sw_, axis=0)
    
    su = np.insert(u1, units_1, np.zeros((units_2, units_1)), axis=1)
    su = np.insert(su, 2*units_1+units_2, np.zeros((units_2, units_1)), axis=1)
    su = np.insert(su, 3*units_1+2*units_2, np.zeros((units_2, units_1)), axis=1)
    su = np.insert(su, 4*units_1+3*units_2, np.zeros((units_2, units_1)), axis=1)
    su_ = np.insert(u2, 0, np.zeros((units_1, units_2)), axis=1)
    su_ = np.insert(su_, units_1+units_2, np.zeros((units_1, units_2)), axis=1)
    su_ = np.insert(su_, 2*units_1+2*units_2, np.zeros((units_1, units_2)), axis=1)
    su_ = np.insert(su_, 3*units_1+3*units_2, np.zeros((units_1, units_2)), axis=1)
    su = np.append(su, su_, axis=0)
    
    sb = np.append(b1[:units_1], b2[:units_2])
    sb = np.append(sb, b1[units_1:units_1*2])
    sb = np.append(sb, b2[units_2:units_2*2])
    sb = np.append(sb, b1[units_1*2:units_1*3])
    sb = np.append(sb, b2[units_2*2:units_2*3])
    sb = np.append(sb, b1[units_1*3:])
    sb = np.append(sb, b2[units_2*3:])
    
    return (sw, su, sb)