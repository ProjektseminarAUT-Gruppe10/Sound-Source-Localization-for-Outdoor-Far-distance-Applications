# Dieses Skript versucht die TDOA im Zeitbereich mit CSOM zu bestimmen

# Imports
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sg

sys.path.append("..\\..\\simulation")
sys.path.append("..\\..\\libraries")
from GeometryLibrary import calculateMicrophoneArray_2
from SimulationLibrary import load_configs, simulate
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, KarstenDOA_calculateSteep_linear_simple
from GeometryLibrary import getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear, KarstenDOA_calculateCurve_nonlinear, getMicrophonePair_DOA_Intersection_linear
from GeometryLibrary import getAngle_angle1, get_tCurve, angle_radians
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import tdoa_csom, centralSecondOrderMomentFunction_part, getSignalPower_UsingTime_AverageFree, tdoa_gcc_phat, centralSecondOrderMomentFunction
from SignalProcessingLibrary import butterWorthFilter, wienerFilter

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
    return 0.58 * math.sin(x * (2 * math.pi * 400.0)) if (x > 0.05 and x < 0.1) else 0

def convertPoint(p):
    return getPoint(p[0],p[1])

def cutSignal(signal, meta_data, time_window):
    time = np.arange(meta_data["number_samples"])/meta_data["sampling_rate"] 
    fromIndex = int(time_window["from"]/meta_data["duration"]*meta_data["number_samples"])
    toIndex   = int(time_window["to"]/meta_data["duration"]*meta_data["number_samples"])    
    return time[fromIndex:toIndex], signal[fromIndex:toIndex]

def cutSignal_sample(signal, meta_data, sample_window):
    time = np.arange(meta_data["number_samples"])/meta_data["sampling_rate"] 
    fromIndex = sample_window["from"]
    toIndex   = sample_window["to"] 
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

def calcSTDOA(signalA, signalB, micPosList, indA, indB):        
    std1 = tdoa_gcc_phat(signalA, signalB, fs=meta["sampling_rate"])[0]*343.3
    rstd1 = getRealDeltaS(source_pos, micPosList[indA], micPosList[indB])
    print("STDOA Estimated / Real: ",std1,rstd1)
   
def getRealDeltaS(source_pos, micAPos, micBPos):
    return distance(micAPos, source_pos) - distance(micBPos, source_pos)

def getRealTDOA(source_pos, micAPos, micBPos):
    return getRealDeltaS(source_pos, micAPos, micBPos) / 343.3

def updateConfig(config, micA, micB, microphone_noise_sigm, noise_environment, noise_source, source_pos):
    config["microphone_noise_mus"] = list()
    config["microphone_noise_sigmas"] = list()
    config["microphone_noise_amplitudes"] = list()
    config["microphone_positions"] = list()
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_sigmas"].append(microphone_noise_sigm)
    config["microphone_noise_sigmas"].append(microphone_noise_sigm)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_positions"].append(micA)
    config["microphone_positions"].append(micB)
    config["source_position"] = source_pos
    config["source_noise_sigma"] = noise_source
    config["general_noise_sigma"] = noise_environment
    return config

def getSNR(signal):
    mx = np.max(signal)
    thrFil = np.asarray([signal > mx*0.9])
    fromIdx = np.where(thrFil == True)[1][0]
    toIdx   = thrFil.shape[1]-np.where(np.flip(thrFil) == True)[1][0]
    time_window_signal  = {"from" : fromIdx, "to": toIdx}
    sig_cut = cutSignal_sample(signal, meta, time_window_signal)   
    time_window_signal  = {"from" : 0, "to": 2000}
    noi_cut = cutSignal_sample(signal, meta, time_window_signal)
    powerSig = getSignalPower_UsingTime_AverageFree(sig_cut[1])
    powerNoi = getSignalPower_UsingTime_AverageFree(noi_cut[1])
    snrFac   = (powerSig-powerNoi)/(powerNoi)
    snrDB    = 10*np.log(snrFac)
#    print(powerSig, powerNoi, snrFac, snrDB)
    return powerSig, powerNoi, snrFac, snrDB

# Lade Konfiguration
config = load_configs("configEXP.json")[0]
N = 20

distances = [1,5,10,20,40,60,80,100]
angles    = [0, 15, 45, 75, 90]

f = open('out-unfiltered.txt', 'w')
print("dis;ang;source_pos;SNR;TDOA_real;TDOA_CSOM;TDOA_GCCP")
print("dis;ang;source_pos;SNR;TDOA_real;TDOA_CSOM;TDOA_GCCP",file=f)

for dis in distances:
    for ang in angles:
        SNRls = list()
        TDOA_realls = list()
        TDOA_csomls = list()
        TDOA_GCCPls = list()
        for n in range(0,N):
            # Adjust Configurations
            mic_dist = 0.3
            micA = getPoint(-mic_dist,0)
            micB = getPoint(mic_dist,0)
            noise_microphone = 0.003
            noise_environment = 0.04
            noise_source = 0.01
            source_pos = getPoint(dis*np.sin(angle_radians(ang)),dis*np.cos(angle_radians(ang)))
            config = updateConfig(config, micA, micB, noise_microphone, noise_environment, noise_source, source_pos)
            
            # Starte Simulation
            loaded = simulate(config, config["source_position"], signal_function)
            signals = loaded.get_measurements()
            meta    = loaded.get_meta_data()
            signalA = signals[0]
            signalB = signals[1]
            signalAF = signalA#butterWorthFilter(wienerFilter(signalA), meta["sampling_rate"], 1500)
            signalBF = signalB#butterWorthFilter(wienerFilter(signalB), meta["sampling_rate"], 1500)
            
            TDOA_CSOM, t, csom = tdoa_csom(signalAF, signalBF, fs=meta["sampling_rate"], window=500)
            TDOA_GCCP, cc = tdoa_gcc_phat(signalAF, signalBF, fs=meta["sampling_rate"])
            TDOA_real = getRealTDOA(source_pos, micA, micB)
            a, b, c, SNR = getSNR(signalAF)
           
            SNRls.append(SNR)
            TDOA_realls.append(TDOA_real)
            TDOA_csomls.append(TDOA_CSOM)
            TDOA_GCCPls.append(TDOA_GCCP)
            
        print(dis, ";", ang, ";", source_pos, ";", np.average(SNRls) , ";", np.std(SNRls), ";", np.average(TDOA_realls) , ";", np.std(TDOA_realls), ";", np.average(TDOA_csomls) , ";", np.std(TDOA_csomls), ";", np.average(TDOA_GCCPls) , ";", np.std(TDOA_GCCPls))
        print(dis, ";", ang, ";", source_pos, ";", np.average(SNRls) , ";", np.std(SNRls), ";", np.average(TDOA_realls) , ";", np.std(TDOA_realls), ";", np.average(TDOA_csomls) , ";", np.std(TDOA_csomls), ";", np.average(TDOA_GCCPls) , ";", np.std(TDOA_GCCPls), file=f)

#        print(dis, ";", ang, ";", source_pos, ";", SNR, ";", TDOA_real, ";", TDOA_CSOM, ";", TDOA_GCCP)
#        print(dis, ";", ang, ";", source_pos, ";", SNR, ";", TDOA_real, ";", TDOA_CSOM, ";", TDOA_GCCP, file=f)
f.close()