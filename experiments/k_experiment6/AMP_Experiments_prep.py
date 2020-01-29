# Dieses Skript versucht die TDOA im Zeitbereich mit CSOM zu bestimmen
# Variation der Mikrofonabstände sowie Sampling Frequenz 96kHz

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

import matplotlib.pyplot as plt


sys.path.append("..\\..\\simulation")
sys.path.append("..\\..\\libraries")
from SimulationLibrary import load_configs, simulate
from GeometryLibrary import getAngle_angle1, angle_radians, getPoint, distance
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import butterWorthFilter, tdoa_csom, centralSecondOrderMomentFunction_part, getSignalPower_UsingTime_AverageFree, tdoa_gcc_phat, centralSecondOrderMomentFunction

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

def updateConfig(config, micList, microphone_noise_sigm, noise_environment, noise_source, source_pos):
    config["microphone_noise_mus"] = list()
    config["microphone_noise_sigmas"] = list()
    config["microphone_noise_amplitudes"] = list()
    config["microphone_positions"] = list()
    for m in micList:
        config["microphone_noise_mus"].append(0)
        config["microphone_noise_sigmas"].append(microphone_noise_sigm)
        config["microphone_noise_amplitudes"].append(1)
        config["microphone_positions"].append(m)
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
    return powerSig, powerNoi, snrFac, snrDB

def getFakeTDOA(tdoa, sample_freq):
    fac = 1
    if(tdoa<0):
        fac = -1
        tdoa = -tdoa
    inc = 0
    while (inc*1/sample_freq<tdoa):
        inc += 1
    return inc*1/sample_freq*fac

def getRealTDOA(source_pos, micAPos, micBPos):
    return getRealDeltaS(source_pos, micAPos, micBPos) / 343.3

def getRealDeltaS(source_pos, micAPos, micBPos):
    return distance(micAPos, source_pos) - distance(micBPos, source_pos)

# Lade Konfiguration
config = load_configs("configEXP.json")[0]
N        = 20
mic_dist = 0.4
num_mic  = 8
xval     = 0.7

distances = [1,5,10,20,40,60,80,100]
angles    = [0, 10, 20, 30, 45, 60, 70, 80, 90]

f = open('results-0.70.txt', 'w')
print("dis ; ang ; K_true ; K_estim ; dist_estim")
print("dis ; ang ; K_true ; K_estim ; dist_estim", file=f)

for ang in angles:
    for dis in distances:
        for n in range(0,N):
            # Adjust Configurations
            micList = list()
            for i in range(0,num_mic):
                micList.append(getPoint(mic_dist*np.sin(angle_radians(360/8*(2-i))),mic_dist*np.cos(angle_radians(360/8*(2-i)))))
            noise_microphone  = 0.003
            noise_environment = 0.04
            noise_source      = 0.01
            source_pos = getPoint(dis*np.sin(angle_radians(ang)),dis*np.cos(angle_radians(ang)))
            config     = updateConfig(config, micList, noise_microphone, noise_environment, noise_source, source_pos)
            
            # Signal Simulation
            loaded = simulate(config, config["source_position"], signal_function)
            signals = loaded.get_measurements()
            meta    = loaded.get_meta_data()
            signalsFiltered = list()
            signalsPower    = list()
            signalsSNR      = list()
            for s in signals:
                sf = butterWorthFilter(s, meta["sampling_rate"], 2000)
                powerSig, powerNoi, snrFac, snrDB = getSNR(sf)
                signalsFiltered.append(sf)
                signalsPower.append(powerSig)
                signalsSNR.append(snrDB)
            
            # Calculate True K
            K_true = 0
            for k in range(0,len(micList)):
                K_true += signalsPower[k]*distance(micList[k], source_pos)*distance(micList[k], source_pos)
            K_true /= len(micList)
            
            # Calculate K estimation
            K_estim_exakt = list()
            K_estim_noise = list()
            for i in range(0,len(micList)):
                for j in range(0, len(micList)):
                    if(i!=j):
                        a = getRealTDOA(source_pos, micList[i], micList[j])
                        b = getFakeTDOA(a, 96000)            
                        K1_exakt,  K2_exakt  = estimateK_Pair(signalsPower[i], signalsPower[j], micList[i], micList[j], a*343.2)
                        K1_noised, K2_noised = estimateK_Pair(signalsPower[i], signalsPower[j], micList[i], micList[j], b*343.2)
                        K_estim_exakt.append(K1_exakt)
                        K_estim_exakt.append(K2_exakt)
                        K_estim_noise.append(K1_noised)
                        K_estim_noise.append(K2_noised)
                        
            # Remove NAN
            K_estim_exakt = [x for x in K_estim_exakt if str(x) != 'nan']
            K_estim_noise = [x for x in K_estim_noise if str(x) != 'nan']
            K_estim_exakt.sort()
            K_estim_noise.sort()            
            K_estim = K_estim_noise[int(len(K_estim_noise)*xval)]
            
            # Distance
            distanceReal = distance(source_pos, getPoint(0,0))
            angleReal    = 90 - angle_degree(getAngle_angle1(getPoint(0,0), source_pos))
            
            # Distance Estimated
            radiusList = list()
            for i in range(0,len(micList)):
                radiusList.append(np.sqrt(K_estim/signalsPower[i]))
            distanceEstim = np.average(radiusList)
            
            print(dis, ";", ang, ";", K_true, ";", K_estim, ";", distanceReal, ";", distanceEstim)
            print(dis, ";", ang, ";", K_true, ";", K_estim, ";", distanceReal, ";", distanceEstim, file=f)
f.close()