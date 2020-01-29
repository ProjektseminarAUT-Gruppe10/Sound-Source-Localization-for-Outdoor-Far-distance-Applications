# Imports
import sys
import math
import numpy as np

from SimulationLibrary import load_configs, simulate

sys.path.append("..\\..\\libraries")
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GeometryLibrary import getAngle_angle1
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree

sys.path.append("..\\..\\tdoa")
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

def plausibleFilter_AMP(solutions):
    if(solutions[0][1]<0):
        return solutions[1]
    else:
        return solutions[0]
    
# Lade Konfiguration
config = load_configs("configAMP.json")[0]


config["source_position"] = [10, 20]

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
FILTER_LOW = int(len(K_list)*0.4)
FILTER_HIGH = len(K_list)-1
K_list = [x for x in K_list if str(x) != 'nan']
K_list.sort()
K_list_filt = [x for x in K_list if x > 0.05] #K_list[FILTER_LOW-1:FILTER_HIGH]
estim_K = np.median(K_list_filt)

import matplotlib.pyplot as plt
#plt.plot(K_list)
#

real_K = 0
for i in range(0,7):
    real_K += distance(convertPoint(config["microphone_positions"][i]), source_pos) * np.sqrt(getSignalPower_UsingTime_AverageFree(np.asarray(signals[i])))
real_K /= 8

print("Real K ",real_K)
print("Estim K ",estim_K)

rA = estim_K/np.sqrt(powerA)
rB = estim_K/np.sqrt(powerB)

estimPoints = list()
for i in range(0,7):
    for j in range(0,7):
        if(i!=j):
            rI = estim_K/np.sqrt(getSignalPower_UsingTime_AverageFree(np.asarray(signals[i])))
            rJ = estim_K/np.sqrt(getSignalPower_UsingTime_AverageFree(np.asarray(signals[j])))
            solutions = getIntersectionPointsCircle(convertPoint(config["microphone_positions"][i]), rI, convertPoint(config["microphone_positions"][j]), rJ)
            if(len(solutions)!=0):
                estimPoint = plausibleFilter_AMP(solutions)
                estimPoints.append(estimPoint)
            
# Ergebnisse auswerten
array_center = getPoint(0,0)
distList = list()
anglList = list()
for p in estimPoints:
    distList.append(distance(array_center,convertPoint(p)))
    anglList.append(getAngle_angle1(array_center, convertPoint(p)))
            
d = np.average(distList)
a = np.average(anglList)
estimPoint = getPoint(d*np.cos(a),d*np.sin(a))

# Plot Results
fig = plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(K_list, label="sorted K list")
plt.plot(np.arange(len(K_list)-len(K_list_filt),len(K_list)),K_list_filt, label="filtered K list")
plt.hlines(estim_K, 0, 80, label="estimated K", colors="red")
plt.hlines(real_K, 0, 80, label="real K", colors="green")
plt.vlines(FILTER_LOW,0,1,label="filter_low")
plt.vlines(FILTER_HIGH,0,1,label="filter_high")
plt.legend(loc=4)

plt.subplot(1,3,2)
for p in config["microphone_positions"]:
    point = convertPoint(p)
    drawPoint(point, ".", "black", 40)    
drawPoint(convertPoint(config["source_position"]), "x", "red", 40)
drawPoint(estimPoint, "v", "green", 40)
for p in estimPoints:
    drawPoint(p, "x", "green", 30)
#drawCircle(micA_pos,rA,"blue")
#drawCircle(micB_pos,rB,"blue")
plt.xlim(-15,15)
plt.ylim(-5,30)
plt.grid()
plt.title("Geometry")
plt.gca().set_aspect('equal', adjustable='box')

plt.subplot(1,3,3)
plt.hist(K_list, bins=8)