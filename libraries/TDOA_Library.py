# -*- coding: utf-8 -*-
import basic_tdoa
import array_default
import numpy as np
import sys

#sys.path.append('..\\libraries\\tdoa')
sys.path.append("..\\..\\tdoa")
import tdoa.basic_tdoa

def calculate_TDOA_Matrix(signals, array_params):
    #array_params = array_parameters.ArrayParameters(positions)
    array_params = array_default.ArrayDefault()
    Tdoa = basic_tdoa.BasicTDOA(signals, 200, 0.5, array_params)
    
    #calculate TDOA matrix
    n = len(signals.measurements)
    TDOA_mat = np.zeros((n,n))
    for mic1 in range(0,n):
        for mic2 in range(0,n):
            delta_sample = (Tdoa.gcc_phat(signals.measurements[mic1], signals.measurements[mic2]))            
            delta_time = signals.meta_data["sampling_spacing"]*delta_sample
            TDOA_mat[mic1][mic2] = delta_time
    return TDOA_mat
