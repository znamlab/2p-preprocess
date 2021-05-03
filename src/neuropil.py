from ast_model import ast_model
import numpy as np

def correct_neuropil(dpath):
    Fr = np.load(dpath + 'F.npy')
    Fn = np.load(dpath + 'Fneu.npy')

    cell = 20
    trace, var_params = ast_model(np.vstack([Fr[cell,:], Fn[cell,:]]),
                            np.array([1, 1]))

    
