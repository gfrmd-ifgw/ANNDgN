import numpy as np
import pandas as pd
from sklearn import preprocessing
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation, ActivityRegularization, Dropout
from keras.models import load_model
import pickle
from keras import backend as K
import tensorflow as tf
import calc_spec
from calc_hvl_nn import *

minimum_dgm = np.loadtxt('NN/trained_mgd/dgm_min_norm_value.txt')

model_vec = []
for i in range(5):
    model = load_model('NN/trained_mgd/trained_model'+str(i)+'.h5')
    model_vec.append(model)
model_vec_kerma = []
for i in range(5):
    with open('NN/trained_kerma/model_kerma_trained'+str(i)+'.pkl', 'rb') as file:
        model = pickle.load(file)
        model_vec_kerma.append(model)

minimum_kerma = np.loadtxt('NN/trained_kerma/minimum_kerma.txt')


with open("NN/trained_mgd/min_max.pkl", 'rb') as file:
    dgm_min_max_scaler = pickle.load(file)

with open("NN/trained_kerma/min_max_kerma.pkl", 'rb') as file:
    kerma_min_max_scaler = pickle.load(file)

def predict_mgd(potential, radius, thickness, gland, skin, adipose):
    energy_vec = np.arange(8.25, potential+0.25, 0.5)
    vec = [0.0, radius, thickness, gland, skin, adipose]
    matrix = np.tile(vec, (len(energy_vec),1))
    matrix[:,0] = energy_vec
    empty_vec = np.arange(1.25, potential+0.25, 0.5)
    matrix = dgm_min_max_scaler.transform(matrix)
    matrix_results = np.zeros((5,(len(empty_vec))))                              
    for idx, model in enumerate(model_vec):
        dgm = model.predict(matrix)
        dgm = dgm*minimum_dgm
        matrix_results[idx,14:] = dgm[:,0]
    return matrix_results


def predict_kerma(potential, thick, compress_dist):
    energy_vec = np.arange(8.25, potential+0.25, 0.5, dtype=float)
    vec = [0.0,thick, compress_dist]
    matrix = np.tile(vec, (len(energy_vec),1))
    matrix[:,0] = energy_vec
    empty_vec = np.arange(1.25, potential+0.25, 0.5)
    matrix = kerma_min_max_scaler.transform(matrix)
    matrix_results = np.zeros((5,(len(empty_vec))))                              
    for idx, model in enumerate(model_vec_kerma):
        kerma = model.predict(matrix)
        kerma = kerma*minimum_kerma
        matrix_results[idx,14:] = kerma
    return matrix_results


def predict_mgd_kerma(potential, radius, thick, gland, skin, adipose, compress_dist):
    mgd = predict_mgd(potential, radius, thick, gland, skin, adipose)
    kerma = predict_kerma(potential, thick, compress_dist)
    return mgd, kerma

def predict_dgn_poli(anode, Filter, thick_filter, potential, radius, thick, gland, skin, adipose, compress_dist, pmma):
    spec = calc_spec.return_df_spec(anode, Filter, potential, thick_filter, pmma)
    mgd, kerma = predict_mgd_kerma(potential, radius, thick, gland, skin, adipose, compress_dist)    
    
    mgd = mgd*spec[1].values
    mgd_spec = np.sum(mgd, axis=1)
    mgd = np.mean(mgd_spec)
    e_mgd = np.std(mgd_spec, ddof=1)
    kerma_n = kerma*spec[1].values
    kerma = kerma*spec[1].values
    kerma_spec = np.sum(kerma, axis=1)
    kerma = np.mean(kerma_spec)
    e_kerma = np.std(kerma_spec, ddof=1)
    dgn = mgd/kerma
    edgn = dgn*np.sqrt((e_mgd/mgd)**2 + (e_kerma/kerma)**2)
    hvl = return_hvl(potential, np.mean(kerma_n, axis=0), kerma)[0]
    return (dgn, edgn, hvl)
    



def predict_mgd_mono(energy, radius, thickness, gland, skin, adipose):
    dgm_vec = []
    input_array = np.array([energy, radius, thickness, gland, skin, adipose])
    input_array = input_array.reshape(1,6)
    input_array = dgm_min_max_scaler.transform(input_array)
    for idx, model in enumerate(model_vec):
        dgm = model.predict(input_array)
        dgm = dgm*minimum_dgm
        dgm_vec.append(dgm)
    return np.mean(dgm_vec), np.std(dgm_vec, ddof=1)


def predict_kerma_mono(energy, thick, compress_dist):
    kerma_vec = []
    input_array = np.array([energy, thick, compress_dist])
    input_array = input_array.reshape(1,3)
    input_array = kerma_min_max_scaler.transform(input_array)
    for idx, model in enumerate(model_vec_kerma):
        kerma = model.predict(input_array)
        kerma = kerma*minimum_kerma
        kerma_vec.append(kerma)
                            
    return np.mean(kerma_vec), np.std(kerma_vec, ddof=1)
        
    
    


def predict_mgd_mono_vec(matrix):
    dgm_vec = []
    input_array = matrix
    input_array = matrix.reshape(-1,6)
    input_array = dgm_min_max_scaler.transform(input_array)
    for idx, model in enumerate(model_vec):
        dgm = model.predict(input_array)
        dgm = dgm*minimum_dgm
        dgm_vec.append(dgm)
                            
    return np.mean(dgm_vec, axis=0), np.std(dgm_vec, ddof=1, axis=0)


def predict_kerma_mono_vec(matrix):
    kerma_vec = []
    input_array = matrix.reshape(-1,3)
    input_array = kerma_min_max_scaler.transform(input_array)
    for idx, model in enumerate(model_vec_kerma):
        kerma = model.predict(input_array)
        kerma = kerma*minimum_kerma
        kerma_vec.append(kerma)
                            
    return np.mean(kerma_vec, axis=0), np.std(kerma_vec, ddof=1, axis=0)
