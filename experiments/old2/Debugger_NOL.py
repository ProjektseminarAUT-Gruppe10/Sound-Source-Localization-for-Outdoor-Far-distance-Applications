# Imports
import sys
import math
import numpy as np

sys.path.append("..\\..\\simulation")

sys.path.append("..\\..\\libraries")
from SimulationLibrary import load_configs, simulate
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GeometryLibrary import get_tCurve
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric
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

# Lade Konfiguration
config = load_configs("config.json")[0]

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
tdoaMAT  = tdoa.tdoa_gcc_phat(0.0)[1][0]
delta_n1 = tdoaMAT[1][0]
delta_n2 = tdoaMAT[3][2]
delta_t1 = delta_n1*meta["sampling_spacing"]
delta_t2 = delta_n2*meta["sampling_spacing"]
delta_s1 = delta_t1*343.2
delta_s2 = delta_t2*343.2

# TDOA LINEAR VERFAHREN
steep1 = KarstenDOA_calculateSteep_linear_simple(distance(micA_pos, micB_pos), delta_s1)
steep2 = KarstenDOA_calculateSteep_linear_simple(distance(micC_pos, micD_pos), delta_s2)
solutions, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micA_pos, micB_pos, micC_pos, micD_pos, steep1, steep2)
estimationLIN = plausibleFilter_TDOA(solutions)

curveA = get_tCurve(micA_pos, micB_pos, delta_s1)
curveB = get_tCurve(micC_pos, micD_pos, delta_s2)
estimationNOL = optimizeIntersectionPoint_nonLinear_numeric(estimationLIN, curveA, curveB)
    
# Kurven zur Darstellung
X_NOL_CURVE1, Y_NOL_CURVE1 = KarstenDOA_calculateCurve_nonlinear(micA_pos, micB_pos, delta_s1, res=0.01, rang=10)
X_LIN_CURVE1, Y_LIN_CURVE1 = KarstenDOA_calculateCurve_linear(micA_pos, micB_pos, delta_s1, res=0.01, rang=10)
X_NOL_CURVE2, Y_NOL_CURVE2 = KarstenDOA_calculateCurve_nonlinear(micC_pos, micD_pos, delta_s2, res=0.01, rang=10)
X_LIN_CURVE2, Y_LIN_CURVE2 = KarstenDOA_calculateCurve_linear(micC_pos, micD_pos, delta_s2, res=0.01, rang=10)

# Plot Data
import matplotlib.pyplot as plt
plt.xlim(-10,10)
plt.ylim(-1,19)
plt.grid()
plt.title("Geometry")
plt.gca().set_aspect('equal', adjustable='box')
drawPoint(estimationNOL, "x", "green", 40)
drawPoint(estimationLIN, "x", "black", 40)
drawPoint(micA_pos, ".", "black", 40)   
drawPoint(micB_pos, ".", "black", 40)     
drawPoint(micC_pos, ".", "black", 40)   
drawPoint(micD_pos, ".", "black", 40)     
drawPoint(source_pos, "x", "red", 40)
plt.plot(X_NOL_CURVE1, Y_NOL_CURVE1)
plt.plot(X_LIN_CURVE1, Y_LIN_CURVE1)
plt.plot(X_NOL_CURVE1, Y_NOL_CURVE1)
plt.plot(X_LIN_CURVE1, Y_LIN_CURVE1)
plt.plot(X_NOL_CURVE2, Y_NOL_CURVE2)
plt.plot(X_LIN_CURVE2, Y_LIN_CURVE2)
plt.plot(X_NOL_CURVE2, Y_NOL_CURVE2)
plt.plot(X_LIN_CURVE2, Y_LIN_CURVE2)