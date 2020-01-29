# Imports
import sys
import math
import numpy as np

from SimulationLibrary import load_configs, simulate

sys.path.append("..\\libraries")
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree

sys.path.append("..\\tdoa")
import array_parameters
import basic_tdoa

def generateTime(sampling_rate, number_samples):
    return np.arange(0,number_samples)/sampling_rate

def getSourceSignal(time):
    source_signal = np.zeros_like(time)
    for t in range(0,time.shape[0]):
        source_signal[t] = signal_function(time[t])
    return source_signal

def signal_function(x): 
    return math.sin(x * (2 * math.pi * 400.0)) if (x > 0.1 and x < 0.3) else 0

def convertPoint(p):
    return getPoint(p[0],p[1])

def plausibleFilter_TDOA(solutions):
    result = list()
    for p in solutions:
        if(p[1]>0):
            result.append(p)
    if(np.linalg.norm(result[0])>np.linalg.norm(result[1])):
        return result[0]
    else:
        return result[1]

# Lade Konfiguration
config = load_configs("config.json")[0]

config["source_position"] = [5, 10]

# Starte Simulation
loaded = simulate(config, config["source_position"], signal_function)
signals = loaded.get_measurements()
meta    = loaded.get_meta_data()
micA_pos = convertPoint(config["microphone_positions"][0])
micB_pos = convertPoint(config["microphone_positions"][1])
micC_pos = convertPoint(config["microphone_positions"][2])
micD_pos = convertPoint(config["microphone_positions"][3])
source_pos = convertPoint(config["source_position"])

# Source Signal
time = generateTime(meta["sampling_rate"],meta["number_samples"])
source_signal = getSourceSignal(time)

# Do Power Calculations
powerA = getSignalPower_UsingTime_AverageFree(np.asarray(signals[0]))
powerB = getSignalPower_UsingTime_AverageFree(np.asarray(signals[1]))
powerC = getSignalPower_UsingTime_AverageFree(np.asarray(signals[2]))
powerD = getSignalPower_UsingTime_AverageFree(np.asarray(signals[3]))
powerSrc = getSignalPower_UsingTime_AverageFree(source_signal)

# Calculate TDOA
arr = array_parameters.ArrayParameters(config["microphone_positions"])
tdoa = basic_tdoa.BasicTDOA(loaded, 0, 0.0, arr)
TDOA_mat = tdoa.tdoa_gcc_phat(0.0)[1][0]
delta_n1 = TDOA_mat[1][0]   # MIC A and MIC B
delta_n2 = TDOA_mat[2][0]   # MIC A and MIC C
delta_n3 = TDOA_mat[3][0]   # MIC A and MIC D
delta_n4 = TDOA_mat[2][1]   # MIC B and MIC C
delta_n5 = TDOA_mat[3][1]   # MIC B and MIC D
delta_n6 = TDOA_mat[3][2]   # MIC C and MIC D
delta_t1 = delta_n1*meta["sampling_spacing"]
delta_t2 = delta_n2*meta["sampling_spacing"]
delta_t3 = delta_n3*meta["sampling_spacing"]
delta_t4 = delta_n4*meta["sampling_spacing"]
delta_t5 = delta_n5*meta["sampling_spacing"]
delta_t6 = delta_n6*meta["sampling_spacing"]
delta_s1 = delta_t1*343.2
delta_s2 = delta_t2*343.2
delta_s3 = delta_t3*343.2
delta_s4 = delta_t4*343.2
delta_s5 = delta_t5*343.2
delta_s6 = delta_t6*343.2

# AMP Verfahren
K_list = list()
R_listA = list()
K1, K2 = estimateK_Pair(powerA, powerB, micA_pos, micB_pos, delta_s1)
K_list.append(K1)
K_list.append(K2)
R_listA.append(K1/np.sqrt(powerA))
R_listA.append(K2/np.sqrt(powerA))
K1, K2 = estimateK_Pair(powerA, powerC, micA_pos, micC_pos, delta_s2)
K_list.append(K1)
K_list.append(K2)
R_listA.append(K1/np.sqrt(powerA))
R_listA.append(K2/np.sqrt(powerA))
K1, K2 = estimateK_Pair(powerA, powerD, micA_pos, micD_pos, delta_s3)
K_list.append(K1)
K_list.append(K2)
R_listA.append(K1/np.sqrt(powerA))
R_listA.append(K2/np.sqrt(powerA))
K1, K2 = estimateK_Pair(powerB, powerC, micB_pos, micC_pos, delta_s4)
K_list.append(K1)
K_list.append(K2)
K1, K2 = estimateK_Pair(powerB, powerD, micB_pos, micD_pos, delta_s5)
K_list.append(K1)
K_list.append(K2)
K1, K2 = estimateK_Pair(powerC, powerD, micC_pos, micD_pos, delta_s6)
K_list.append(K1)
K_list.append(K2)

K_list.sort()

import matplotlib.pyplot as plt
plt.plot(K_list)

real_K = distance(micA_pos, source_pos) * np.sqrt(powerA)
print("Real K ",real_K)
real_K = distance(micB_pos, source_pos) * np.sqrt(powerB)
print("Real K ",real_K)
real_K = distance(micC_pos, source_pos) * np.sqrt(powerC)
print("Real K ",real_K)
real_K = distance(micD_pos, source_pos) * np.sqrt(powerD)
print("Real K ",real_K)

print(" ")
print(" ")

print(distance(micA_pos, source_pos), "\t", real_K/np.sqrt(powerA))
print(distance(micB_pos, source_pos), "\t", real_K/np.sqrt(powerB))
print(distance(micC_pos, source_pos), "\t", real_K/np.sqrt(powerC))
print(distance(micD_pos, source_pos), "\t", real_K/np.sqrt(powerD))
    
rA_1 = K1/np.sqrt(powerA)
rA_2 = K2/np.sqrt(powerA)
rB_1 = K1/np.sqrt(powerB)
rB_2 = K2/np.sqrt(powerB)

for p in config["microphone_positions"]:
    point = convertPoint(p)
    drawPoint(point, ".", "black", 40)    
drawPoint(convertPoint(config["source_position"]), "x", "red", 40)
drawCircle(micA_pos,rA_1,"blue")
drawCircle(micA_pos,rA_2,"red")
drawCircle(micB_pos,rB_1,"blue")
drawCircle(micB_pos,rB_2,"red")
plt.xlim(-10,10)
plt.ylim(0,20)
plt.grid()
plt.title("Geometry")
plt.gca().set_aspect('equal', adjustable='box')
