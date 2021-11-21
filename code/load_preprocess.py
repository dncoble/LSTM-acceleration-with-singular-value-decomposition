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
    
    
    #%% preprocess
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
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(X, y, acc_t_reshape)
    X_train = np.expand_dims(X_train, 2)
    X_test = np.expand_dims(X_test, 2)
    y_train = np.expand_dims(y_train, 2)
    y_test = np.expand_dims(y_test, 2)
    t_train = np.expand_dims(t_train, 2)
    t_test = np.expand_dims(t_test, 2)
    
    
    sav_X_train = open('./pickles/X_train', 'wb')
    sav_y_train = open('./pickles/y_train', 'wb')
    sav_t_train = open('./pickles/t_train', 'wb')
    sav_X_test = open('./pickles/X_test', 'wb')
    sav_y_test = open('./pickles/y_test', 'wb')
    sav_t_test = open('./pickles/t_test', 'wb')
    
    pickle.dump(X_train, sav_X_train)
    pickle.dump(y_train, sav_y_train)
    pickle.dump(t_train, sav_t_train)
    pickle.dump(X_test, sav_X_test)
    pickle.dump(y_test, sav_y_test)
    pickle.dump(t_test, sav_t_test)
    
    joblib.dump(acc_scaler, './pickles/acc_scaler', compress=True)
    joblib.dump(pin_scaler, './pickles/pin_scaler', compress=True)
    
    sav_X_train.close()
    sav_y_train.close()
    sav_t_train.close()
    sav_X_test.close()
    sav_y_test.close()
    sav_t_test.close()
# i don't think this actually works properly, but i'll fix it if i ever need it
def save_model_weights_as_json(model, savpath="model_weights.json"):
    import json
    i = 0
    data = {}
    layer_weights = []
    for layer in model.layers:
        layer_weights = [a.tolist() for a in layer.get_weights()]
        data["layer" + str(i)] = layer_weights
        i += 1
    with open(savpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

# save model weights as folder of csvs.
def save_model_weights_as_csv(model, savpath = "./model_weights"):
    import os
    import os.path as path
    from numpy import savetxt
    if(not path.exists(savpath)):
        os.mkdir(savpath)
    for layer in model.layers[:-1]: # the final layer is a dense top
        layer_path = savpath + "./" + layer.name + "./"
        if(not path.exists(layer_path)):
            os.mkdir(layer_path)
        W, U, b = layer.get_weights()
        units = U.shape[0]
        
        savetxt(layer_path+"Wi.csv",W[:,:units],delimiter=',')
        savetxt(layer_path+"Wf.csv",W[:,units:units*2],delimiter=',')
        savetxt(layer_path+"Wc.csv",W[:,units*2:units*3],delimiter=',')
        savetxt(layer_path+"Wo.csv",W[:,units*3:],delimiter=',')
        savetxt(layer_path+"Ui.csv",U[:,:units],delimiter=',')
        savetxt(layer_path+"Uf.csv",U[:,units:units*2],delimiter=',')
        savetxt(layer_path+"Uc.csv",U[:,units*2:units*3],delimiter=',')
        savetxt(layer_path+"Uo.csv",U[:,units*3:],delimiter=',')
        savetxt(layer_path+"bi.csv",b[:units],delimiter=',')
        savetxt(layer_path+"bf.csv",b[units:units*2],delimiter=',')
        savetxt(layer_path+"bc.csv",b[units*2:units*3],delimiter=',')
        savetxt(layer_path+"bo.csv",b[units*3:],delimiter=',')
    
    #save dense top layer
    dense_top = model.layers[-1]
    in_weights, out_weights = dense_top.get_weights()
    layer_path = savpath + "./dense_top./"
    if(not path.exists(layer_path)):
        os.mkdir(layer_path)    
    savetxt(layer_path+"in_weights.csv",in_weights,delimiter=',')
    savetxt(layer_path+"out_weights.csv",out_weights,delimiter=',')

# with LabVIEW it is easiest if rather than a json file I have multiple csvs.
# savpath should be a folder
def json_to_csv(json_file, savpath):
    import os
    import os.path as path
    import json
    import numpy as np
    from numpy import savetxt
    f = open(json_file)
    data = json.load(f)
    f.close()
    if(not path.exists(savpath)):
        os.mkdir(savpath)
    for name, dataset in data.items():
        if(type(dataset) == type([])):
            savetxt(savpath +'/' + name + '.csv', np.array(dataset), delimiter=',')
# actually I want the preprocessed data into csv. i'll only do the test
# datasets though. savpaths should be a folder
def preprocessed_to_csv(savpath):
    import pickle
    import numpy as np
    from numpy import savetxt
    import os
    from os import path
    if(not path.exists(savpath)):
        os.mkdir(savpath)
    load_X_test = open("./pickles/X_test", 'rb')
    load_y_test = open("./pickles/y_test", 'rb')
    load_t_test = open("./pickles/t_test", 'rb')
    X_test = np.squeeze(pickle.load(load_X_test))
    y_test = np.squeeze(pickle.load(load_y_test))
    t_test = np.squeeze(pickle.load(load_t_test))
    load_X_test.close()
    load_y_test.close()
    load_t_test.close()
    savetxt(savpath+'/X_test.csv',X_test,delimiter=',')
    savetxt(savpath+'/y_test.csv',y_test,delimiter=',')
    savetxt(savpath+'/t_test.csv',t_test,delimiter=',')


if __name__ == '__main__':
    preprocess()