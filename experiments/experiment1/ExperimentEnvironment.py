# Imports
import sys
import math
import numpy as np

sys.path.append("..\\..\\simulation")
sys.path.append("..\\..\\libraries")
from GeometryLibrary import calculateMicrophoneArray_2
from SimulationLibrary import load_configs, simulate
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GeometryLibrary import getAngle_angle1, get_tCurve
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree

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
    return 0.58 * math.sin(x * (2 * math.pi * 400.0)) if (x > 0.05 and x < 0.4) else 0

def convertPoint(p):
    return getPoint(p[0],p[1])

def cutSignal(signal, meta_data, time_window):
    time = np.arange(meta_data["number_samples"])/meta_data["sampling_rate"] 

    fromIndex = int(time_window["from"]/meta_data["duration"]*meta_data["number_samples"])
    toIndex   = int(time_window["to"]/meta_data["duration"]*meta_data["number_samples"])
    
    return time[fromIndex:toIndex], signal[fromIndex:toIndex]

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
    
def SSL_TDOA_LIN(s_tdoaMAT, micPosList, indA, indB, indC, indD):        
    std1 = s_tdoaMAT[indB][indA]
    std2 = s_tdoaMAT[indD][indC]
    rstd1 = getRealDeltaS(source_pos, micPosList[indA], micPosList[indB])
    rstd2 = getRealDeltaS(source_pos, micPosList[indC], micPosList[indD])
    print("STDOA 1: ",std1,rstd1)
    print("STDOA 2: ",std2,rstd2)
    steep1 = KarstenDOA_calculateSteep_linear_simple(distance(micList[indA], micList[indB]), s_tdoaMAT[indB][indA])
    steep2 = KarstenDOA_calculateSteep_linear_simple(distance(micList[indC], micList[indD]), s_tdoaMAT[indD][indC])
    solutions, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micList[indA], micList[indB], micList[indC], micList[indD], steep1, steep2)
    estimation = plausibleFilter_TDOA(solutions)
    return estimation

#def SSL_TDOA_LIN(signals, micPosList, indA, indB, indC, indD):          
#    delta_tX1 = gcc_phat(np.asarray(signals[indA]), np.asarray(signals[indB]), fs=meta["sampling_rate"])[0]
#    delta_tX2 = gcc_phat(np.asarray(signals[indC]), np.asarray(signals[indD]), fs=meta["sampling_rate"])[0]
#    delta_sX1 = delta_tX1*343.3
#    delta_sX2 = delta_tX2*343.3
#    
#    std1 = delta_sX1#s_tdoaMAT[indB][indA]
#    std2 = delta_sX2#s_tdoaMAT[indD][indC]
#    rstd1 = getRealDeltaS(source_pos, micPosList[indA], micPosList[indB])
#    rstd2 = getRealDeltaS(source_pos, micPosList[indC], micPosList[indD])
#    
#    print("STDOA 1: ",std1,rstd1)
#    print("STDOA 2: ",std2,rstd2)
#    steep1 = KarstenDOA_calculateSteep_linear_simple(distance(micList[indA], micList[indB]), s_tdoaMAT[indB][indA])
#    steep2 = KarstenDOA_calculateSteep_linear_simple(distance(micList[indC], micList[indD]), s_tdoaMAT[indD][indC])
#    solutions, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micList[indA], micList[indB], micList[indC], micList[indD], steep1, steep2)
#    estimation = plausibleFilter_TDOA(solutions)
#    return estimation

def SSL_TDOA_NOL(s_tdoaMAT, micList, indA, indB, indC, indD, estimationLIN):    
    curveA = get_tCurve(micList[indA], micList[indB], s_tdoaMAT[indB][indA])
    curveB = get_tCurve(micList[indC], micList[indD], s_tdoaMAT[indD][indC])
    estimationNOL = optimizeIntersectionPoint_nonLinear_numeric(estimationLIN, curveA, curveB)
    return estimationNOL
   
def getRealDeltaS(source_pos, micAPos, micBPos):
    return distance(micAPos, source_pos) - distance(micBPos, source_pos)

# Lade Konfiguration
config = load_configs("configEXP.json")[0]

# Setze Mikrofon Positionen
micList = calculateMicrophoneArray_2(array_radius=0.5, array_center=getPoint(0,0))
config["microphone_noise_mus"] = list()
config["microphone_noise_sigmas"] = list()
config["microphone_noise_amplitudes"] = list()
config["microphone_positions"] = list()
for m in micList:
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_sigmas"].append(0)#0.003)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_positions"].append(m)
    
# Setze Tonquelle
config["source_position"] = [10, 10]
source_pos = convertPoint(config["source_position"])

# Starte Simulation
loaded = simulate(config, config["source_position"], signal_function)
signals = loaded.get_measurements()
meta    = loaded.get_meta_data()

# Berechne TDOAs
arr = array_parameters.ArrayParameters(config["microphone_positions"])
tdoa = basic_tdoa.BasicTDOA(loaded, 0, 0, arr)
n_tdoaMAT = np.asarray(tdoa.tdoa_gcc_phat(0.0)[1][0])
t_tdoaMAT = n_tdoaMAT*meta["sampling_spacing"]
s_tdoaMAT = t_tdoaMAT*343.2

# SSL - TDOA_LIN
estimationLIN = SSL_TDOA_LIN(s_tdoaMAT, micList, 7, 0, 0, 1)

# SSL - TDOA_NOL
estimationNOL = SSL_TDOA_NOL(s_tdoaMAT, micList, 7, 0, 0, 1, estimationLIN)

# Schneide Zeitfenster auf relevante Bereiche
time_window_noise   = {"from" : 0.01, "to": 0.05}
time_window_signal  = {"from" : 0.3, "to": 0.4}
signalsCutNoise = list()
signalsCutSignal = list()
for sig in signals:
    signalsCutNoise.append(cutSignal(np.asarray(sig),meta,time_window_noise))
    signalsCutSignal.append(cutSignal(np.asarray(sig),meta,time_window_signal))
    
# Berechne Signalleistungen
powerNoise = list()
powerSignal = list()
for sig in signalsCutNoise:
    powerNoise.append(getSignalPower_UsingTime_AverageFree(sig[1]))
counter = 0
for sig in signalsCutSignal:
    powerSignal.append(getSignalPower_UsingTime_AverageFree(sig[1])-powerNoise[counter])
    counter += 1

# Plot Data
import matplotlib.pyplot as plt
plt.figure(figsize=(20,10))

# Zeichne Geometrie
plt.subplot(1,2,2)
plt.xlim(-10,10)
plt.ylim(-1,19)
plt.grid()
plt.title("Geometry")
plt.gca().set_aspect('equal', adjustable='box')
for m in micList:
    drawPoint(m, ".", "black", 40)
drawPoint(source_pos, "x", "red", 40)
drawPoint(estimationLIN, "x", "blue", 40)
drawPoint(estimationNOL, "x", "blue", 40)

#plt.subplot(4,2,2)
#plt.plot(signals[7])
#plt.subplot(4,2,4)
#plt.plot(signals[0])
#plt.subplot(4,2,6)
#plt.plot(signals[0])
#plt.subplot(4,2,8)
#plt.plot(signals[1])

# Zeichne Quell Signal
time = generateTime(meta["sampling_rate"],meta["number_samples"])
source_signal = getSourceSignal(time)
time_window_signal  = {"from" : 0.05, "to": 0.4}
source_signal_cut = cutSignal(source_signal, meta, time_window_signal)
plt.subplot(9,2,1)
plt.title("source signal ("+"%.4f" %(getSignalPower_UsingTime_AverageFree(source_signal_cut[1])*1000)+" mW)")
plt.plot(time, source_signal)
plt.plot(source_signal_cut[0],source_signal_cut[1])

# Zeichne Rausch Fenster
counter = 1+4
channel = 1
for sig in signalsCutNoise:
    plt.subplot(9,4,counter)
    plt.title("noise ("+"%.4f" %(powerNoise[channel-1]*1000)+" mW)")
    plt.plot(sig[0],sig[1])
    plt.ylabel("Ch "+str(channel))
    counter += 4
    channel += 1

# Zeichne Signal Fenster
counter = 2+4
channel = 1
for sig in signalsCutSignal:
    plt.subplot(9,4,counter)
    plt.title("signal ("+"%.4f" %(powerSignal[channel-1]*1000)+" mW)")
    plt.plot(sig[0],sig[1])
    counter += 4
    channel += 1
    
plt.tight_layout()