# Imports
import sys
import math
import numpy as np

sys.path.append("..\\..\\simulation")
sys.path.append("..\\..\\libraries")
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree
from SimulationLibrary import load_configs, simulate

sys.path.append("..\\..\\tdoa")
import array_parameters
import basic_tdoa

def gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    
    # make sure the length for the FFT is larger or equal than len(sig) + len(refsig)
    n = sig.shape[0] + refsig.shape[0]

    # Generalized Cross Correlation Phase Transform
    SIG = np.fft.rfft(sig, n=n)
    REFSIG = np.fft.rfft(refsig, n=n)
    R = SIG * np.conj(REFSIG)

    cc = np.fft.irfft(R / np.abs(R), n=(interp * n))

    max_shift = int(interp * n / 2)
    if max_tau:
        max_shift = np.minimum(int(interp * fs * max_tau), max_shift)

    cc = np.concatenate((cc[-max_shift:], cc[:max_shift+1]))

    # find max cross correlation index
    shift = np.argmax(np.abs(cc)) - max_shift

    tau = shift / float(fs)
    
    return tau, cc

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
    if(len(result)>1):
        if(np.linalg.norm(result[0])>np.linalg.norm(result[1])):
            return result[0]
        else:
            return result[1]
    elif(len(result)==1):
        return result[0]
    else:
        return getPoint(0,0)
    
import matplotlib.pyplot as plt
for x in range(0,20):
    # Lade Konfiguration
    config = load_configs("config.json")[0]
    config["source_position"] = [-10+x, 5]
    
    # Starte Simulation
    loaded = simulate(config, config["source_position"], signal_function)
    signals = loaded.get_measurements()
    meta    = loaded.get_meta_data()
    micA_pos = convertPoint(config["microphone_positions"][0])
    micB_pos = convertPoint(config["microphone_positions"][1])
    micC_pos = convertPoint(config["microphone_positions"][2])
    micD_pos = convertPoint(config["microphone_positions"][3])
    source_pos = convertPoint(config["source_position"])
    
    # Calculate TDOA
    arr = array_parameters.ArrayParameters(config["microphone_positions"])
    tdoa = basic_tdoa.BasicTDOA(loaded, 0, 0, arr)
    tdoaMAT  = tdoa.tdoa_gcc_phat(0.0)[1][0]
    delta_n1 = tdoaMAT[1][0]
    delta_n2 = tdoaMAT[3][2]
    delta_t1 = delta_n1*meta["sampling_spacing"]
    delta_t2 = delta_n2*meta["sampling_spacing"]
    delta_s1 = delta_t1*343.2
    delta_s2 = delta_t2*343.2
#    
    delta_tX1 = gcc_phat(np.asarray(signals[0]), np.asarray(signals[1]), fs=meta["sampling_rate"])[0]
    delta_tX2 = gcc_phat(np.asarray(signals[2]), np.asarray(signals[3]), fs=meta["sampling_rate"])[0]
    delta_sX1 = delta_tX1*343.2
    delta_sX2 = delta_tX2*343.2
    
    
    true_s1 = distance(source_pos, micA_pos) - distance(source_pos, micB_pos)
    true_s2 = distance(source_pos, micC_pos) - distance(source_pos, micD_pos)
    print(true_s1, delta_s1, true_s2, delta_s2, delta_sX1, delta_sX2)
#    print(true_s1, true_s2, delta_sX1, delta_sX2)

#    delta_s1 = delta_sX1
#    delta_s2 = delta_sX2
    
    # TDOA LINEAR VERFAHREN
    steep1 = KarstenDOA_calculateSteep_linear_simple(distance(micA_pos, micB_pos), delta_s1)
    steep2 = KarstenDOA_calculateSteep_linear_simple(distance(micC_pos, micD_pos), delta_s2)
    solutions, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micA_pos, micB_pos, micC_pos, micD_pos, steep1, steep2)
    estimation = plausibleFilter_TDOA(solutions)
      
    # Kurven zur Darstellung
    X_NOL_CURVE1, Y_NOL_CURVE1 = KarstenDOA_calculateCurve_nonlinear(micA_pos, micB_pos, delta_s1, res=0.01, rang=10)
    X_LIN_CURVE1, Y_LIN_CURVE1 = KarstenDOA_calculateCurve_linear(micA_pos, micB_pos, delta_s1, res=0.01, rang=10)
    X_NOL_CURVE2, Y_NOL_CURVE2 = KarstenDOA_calculateCurve_nonlinear(micC_pos, micD_pos, delta_s2, res=0.01, rang=10)
    X_LIN_CURVE2, Y_LIN_CURVE2 = KarstenDOA_calculateCurve_linear(micC_pos, micD_pos, delta_s2, res=0.01, rang=10)
    
    plt.subplot(4,2,1)
    plt.plot(signals[0])
    plt.subplot(4,2,3)
    plt.plot(signals[1])
    plt.subplot(4,2,5)
    plt.plot(signals[2])
    plt.subplot(4,2,7)
    plt.plot(signals[3])
    
    # Plot Data
    plt.subplot(1,2,2)
    plt.xlim(-10,10)
    plt.ylim(-1,19)
    plt.grid()
    plt.title("Geometry")
    plt.gca().set_aspect('equal', adjustable='box')
    #drawPoint(estimation, "x", "green", 40)
    drawPoint(micA_pos, ".", "black", 40)   
    drawPoint(micB_pos, ".", "black", 40)     
    drawPoint(micC_pos, ".", "black", 40)   
    drawPoint(micD_pos, ".", "black", 40)     
    drawPoint(source_pos, "x", "red", 40)
    plt.plot(X_NOL_CURVE1, Y_NOL_CURVE1)
    plt.plot(X_LIN_CURVE1, Y_LIN_CURVE1)
    plt.plot(X_NOL_CURVE2, Y_NOL_CURVE2)
    plt.plot(X_LIN_CURVE2, Y_LIN_CURVE2)

    plt.pause(0.05)
    plt.clf()