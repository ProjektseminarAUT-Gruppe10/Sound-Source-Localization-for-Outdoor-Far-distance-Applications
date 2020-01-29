# Imports
import sys
import math
import numpy as np

from SimulationLibrary import load_configs, simulate

sys.path.append("..\\libraries")
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear
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

# Lade Konfiguration
config = load_configs("config.json")[0]

# Starte Simulation
loaded = simulate(config, config["source_position"], signal_function)
signals = loaded.get_measurements()
meta    = loaded.get_meta_data()
micA_pos = convertPoint(config["microphone_positions"][0])
micB_pos = convertPoint(config["microphone_positions"][1])
source_pos = convertPoint(config["source_position"])

# Source Signal
time = generateTime(meta["sampling_rate"],meta["number_samples"])
source_signal = getSourceSignal(time)

# Do Power Calculations
powerA = getSignalPower_UsingTime_AverageFree(np.asarray(signals[0]))
powerB = getSignalPower_UsingTime_AverageFree(np.asarray(signals[1]))
powerSrc = getSignalPower_UsingTime_AverageFree(source_signal)

# Calculate TDOA
arr = array_parameters.ArrayParameters(config["microphone_positions"])
tdoa = basic_tdoa.BasicTDOA(loaded, 0, 0.0, arr)
delta_n = tdoa.tdoa_gcc_phat(0.0)[1][0][1][0]
delta_t = delta_n*meta["sampling_spacing"]
delta_s = delta_t*343.2

# TDOA LINEAR VERFAHREN
X_NOL_CURVE, Y_NOL_CURVE = KarstenDOA_calculateCurve_nonlinear(micA_pos, micB_pos, delta_s, res=0.01, rang=10)
X_LIN_CURVE, Y_LIN_CURVE = KarstenDOA_calculateCurve_linear(micA_pos, micB_pos, delta_s, res=0.01, rang=10)

# Plot Data
import matplotlib.pyplot as plt

plt.xlim(-10,10)
plt.ylim(0,20)
plt.grid()
plt.title("Geometry")
plt.gca().set_aspect('equal', adjustable='box')
drawPoint(micA_pos, ".", "black", 40)   
drawPoint(micB_pos, ".", "black", 40)     
drawPoint(source_pos, "x", "red", 40)
plt.plot(X_NOL_CURVE, Y_NOL_CURVE)
plt.plot(X_LIN_CURVE, Y_LIN_CURVE)

## Plot Data
#import matplotlib.pyplot as plt
#plt.subplot(2,2,1)
#plt.plot(time, signals[0])
#plt.ylabel("Microphone 1")
#plt.xlim(0,0.2)
#plt.subplot(2,2,3)
#plt.plot(time, signals[1])
#plt.ylabel("Microphone 2")
#plt.xlim(0,0.2)
#
#plt.subplot(2,2,2)
#for p in config["microphone_positions"]:
#    point = convertPoint(p)
#    drawPoint(point, ".", "black", 40)    
#drawPoint(convertPoint(config["source_position"]), "x", "red", 40)
#drawPoint(intCirc,"v","blue",40)
#drawCircle(micA_pos,rA_1,"blue")
#drawCircle(micA_pos,rA_2,"red")
#drawCircle(micB_pos,rB_1,"blue")
#drawCircle(micB_pos,rB_2,"red")
#plt.xlim(-10,10)
#plt.ylim(0,20)
#plt.grid()
#plt.title("Geometry")
#plt.gca().set_aspect('equal', adjustable='box')
#
#plt.subplot(2,2,4)
#plt.plot(time, source_signal)
#plt.xlim(0,0.2)
#plt.title("Sound Source Signal")











# AMPLITUDE BASED VERFAHREN
#K1, K2 = estimateK_Pair(powerA, powerB, micA_pos, micB_pos, delta_s)
#rA_1 = K1/np.sqrt(powerA)
#rA_2 = K2/np.sqrt(powerA)
#rB_1 = K1/np.sqrt(powerB)
#rB_2 = K2/np.sqrt(powerB)
#estimatedPoint = getIntersectionPointsCircle(micA_pos, rA_2, micB_pos, rB_2)

## Plot Data
#import matplotlib.pyplot as plt
#plt.subplot(2,2,1)
#plt.plot(time, signals[0])
#plt.ylabel("Microphone 1")
#plt.xlim(0,0.2)
#plt.subplot(2,2,3)
#plt.plot(time, signals[1])
#plt.ylabel("Microphone 2")
#plt.xlim(0,0.2)
#
#plt.subplot(2,2,2)
#for p in config["microphone_positions"]:
#    point = convertPoint(p)
#    drawPoint(point, ".", "black", 40)    
#drawPoint(convertPoint(config["source_position"]), "x", "red", 40)
#drawPoint(intCirc,"v","blue",40)
#drawCircle(micA_pos,rA_1,"blue")
#drawCircle(micA_pos,rA_2,"red")
#drawCircle(micB_pos,rB_1,"blue")
#drawCircle(micB_pos,rB_2,"red")
#plt.xlim(-10,10)
#plt.ylim(0,20)
#plt.grid()
#plt.title("Geometry")
#plt.gca().set_aspect('equal', adjustable='box')
#
#plt.subplot(2,2,4)
#plt.plot(time, source_signal)
#plt.xlim(0,0.2)
#plt.title("Sound Source Signal")