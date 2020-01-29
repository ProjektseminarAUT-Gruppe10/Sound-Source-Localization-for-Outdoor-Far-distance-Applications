# Imports
import sys
import math
import numpy as np

from SimulationLibrary import load_configs, simulate

sys.path.append("..\\libraries")
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GeometryLibrary import getAngle_angle1
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
    return 0.58 * math.sin(x * (2 * math.pi * 400.0)) if (x > 0.05 and x < 0.4) else 0

def convertPoint(p):
    return getPoint(p[0],p[1])

def cutSignal(signal, meta_data, time_window):
    time = np.arange(meta_data["number_samples"])/meta_data["sampling_rate"] 

    fromIndex = int(time_window["from"]/meta_data["duration"]*meta_data["number_samples"])
    toIndex   = int(time_window["to"]/meta_data["duration"]*meta_data["number_samples"])
    
    return time[fromIndex:toIndex], signal[fromIndex:toIndex]

    
# Lade Konfiguration
config = load_configs("configSNR.json")[0]


config["source_position"] = [0, 40]

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

time_window_noise   = {"from" : 0.01, "to": 0.5}
time_window_signal  = {"from" : 0.3, "to": 0.4}
signal0 = cutSignal(np.asarray(signals[0]),meta,time_window_signal)
signal1 = cutSignal(np.asarray(signals[1]),meta,time_window_signal)
signal2 = cutSignal(np.asarray(signals[2]),meta,time_window_signal)
signal3 = cutSignal(np.asarray(signals[3]),meta,time_window_signal)

# Do Power Calculations
powerA = getSignalPower_UsingTime_AverageFree(signal0[1])
powerB = getSignalPower_UsingTime_AverageFree(signal1[1])
powerC = getSignalPower_UsingTime_AverageFree(signal2[1])
powerD = getSignalPower_UsingTime_AverageFree(signal3[1])
powerSrc = getSignalPower_UsingTime_AverageFree(source_signal)

print(powerA, "\n", powerB, "\n", powerC, "\n", powerD)

import matplotlib.pyplot as plt
plt.subplot(5,1,1)
plt.plot(source_signal)
plt.subplot(5,1,2)
plt.plot(signal0[0],signal0[1])
plt.subplot(5,1,3)
plt.plot(signal1[0],signal1[1])
plt.subplot(5,1,4)
plt.plot(signal2[0],signal2[1])
plt.subplot(5,1,5)
plt.plot(signal3[0],signal3[1])