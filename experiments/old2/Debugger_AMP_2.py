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

config["source_position"] = [1, 20]

config["microphone_positions"].append([-0.6, -0.3])
config["microphone_positions"].append([-0.2, -0.6])
config["microphone_positions"].append([+0.2, -0.6])
config["microphone_positions"].append([+0.6, -0.3])
config["microphone_noise_amplitudes"].append(1)
config["microphone_noise_amplitudes"].append(1)
config["microphone_noise_amplitudes"].append(1)
config["microphone_noise_amplitudes"].append(1)
config["microphone_noise_mus"].append(0)
config["microphone_noise_mus"].append(0)
config["microphone_noise_mus"].append(0)
config["microphone_noise_mus"].append(0)
config["microphone_noise_sigmas"].append(0)
config["microphone_noise_sigmas"].append(0)
config["microphone_noise_sigmas"].append(0)
config["microphone_noise_sigmas"].append(0)

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

K_list = list()

for i in range(0,7):
    for j in range(0,7):
        if(i!=j):
            delta_n1 = TDOA_mat[j][i]   # MIC A and MIC B
            delta_t1 = delta_n1*meta["sampling_spacing"]
            delta_s1 = delta_t1*343.2
            powerA = getSignalPower_UsingTime_AverageFree(np.asarray(signals[i]))
            powerB = getSignalPower_UsingTime_AverageFree(np.asarray(signals[j]))
            micA_pos = convertPoint(config["microphone_positions"][i])
            micB_pos = convertPoint(config["microphone_positions"][j])
            K1, K2 = estimateK_Pair(powerA, powerB, micA_pos, micB_pos, delta_s1)
            K_list.append(K1)
            K_list.append(K2)
            
# AMP Verfahren
K_list = [x for x in K_list if str(x) != 'nan']
K_list.sort()
K_list_filt = K_list[int(len(K_list)/13*6):len(K_list)-1]
estim_K = np.median(K_list_filt)

import matplotlib.pyplot as plt
#plt.plot(K_list)
#

real_K = distance(micA_pos, source_pos) * np.sqrt(powerA)

print("Real K ",real_K)
print("Estim K ",estim_K)

rA = estim_K/np.sqrt(powerA)
rB = estim_K/np.sqrt(powerB)


fig = plt.figure(figsize=(16,8))

plt.subplot(1,2,1)
plt.plot(K_list, label="sorted K list")
plt.plot(np.arange(int(len(K_list)/13*6),len(K_list)-1),K_list_filt, label="filtered K list")
plt.hlines(estim_K, 0, 80, label="estimated K")
plt.hlines(estim_K, 0, 80, label="real K")
plt.legend(loc=4)

plt.subplot(1,2,2)
for p in config["microphone_positions"]:
    point = convertPoint(p)
    drawPoint(point, ".", "black", 40)    
drawPoint(convertPoint(config["source_position"]), "x", "red", 40)
drawCircle(micA_pos,rA,"blue")
drawCircle(micB_pos,rB,"blue")
plt.xlim(-15,15)
plt.ylim(-5,30)
plt.grid()
plt.title("Geometry")
plt.gca().set_aspect('equal', adjustable='box')
