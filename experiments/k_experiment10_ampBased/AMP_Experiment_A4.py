# Imports
import sys
import math
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.io.wavfile import read

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
sys.path.append("..\\..\\simulation")
sys.path.append("..\\..\\libraries")
from SimulationLibrary import load_configs, simulate
from GeometryLibrary import getAngle_angle1, angle_radians, getPoint, distance
from OptimizationLibrary import optimizeIntersectionPoint_nonLinear_numeric
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import butterWorthFilter, tdoa_csom, centralSecondOrderMomentFunction_part, getSignalPower_UsingTime_AverageFree, tdoa_gcc_phat, centralSecondOrderMomentFunction

# Methods
def loadTimeWindows():
    data = genfromtxt('CutResults.csv', delimiter=';',dtype="str")    
    keyList = list()
    limitList = list()    
    for i in range(1,data.shape[0]):
        key = str(data[i][0])+","+str(data[i][1])+","+str(data[i][2])+","
        if(str(data[i][3])=="0"):
            key += "00"
        else:
            key += str(data[i][3]).replace('.',',')
        keyList.append(key)        
        lims = list()
        fromN = float(data[i][4])
        toN   = float(data[i][5])
        fromS = float(data[i][6])
        toS   = float(data[i][7])
        lims.append(fromN)
        lims.append(toN)
        lims.append(fromS)
        lims.append(toS)
        limitList.append(lims)        
    return keyList, limitList

def load_measurement(array, soundSource, angle, dist):    
    meta_data = {}
    data = read("../../measurements/measurement_3/"+array+"_"+soundSource+"_"+angle+"_"+dist+"-01.wav")
    meta_data["sampling_rate"]    = data[0]                         # Sampling Rate in Hz
    meta_data["duration"]         = data[1].shape[0]/data[0]        # Duration in seconds
    meta_data["sampling_spacing"] = 1/meta_data["sampling_rate"]    # Time distance between two samples
    meta_data["number_samples"]   = data[1].shape[0]                # Total number of samples    
    data = np.array(data[1],dtype=float)
    data = np.reshape(data,(data.shape[0],1))
    for i in (1,2,3,4,5,6,7,8):
        b = read("../../measurements/measurement_3/"+array+"_"+soundSource+"_"+angle+"_"+dist+"-0"+str(i)+".wav")
        b = np.array(b[1],dtype=float)
        b = np.reshape(b,(b.shape[0],1))
        data = np.concatenate((data,b),axis=1)
    data = np.transpose(data)    
    return data, meta_data

def cutSignal(signal, meta_data, time_window):
    time = np.arange(meta_data["number_samples"])/meta_data["sampling_rate"] 
    fromIndex = int(time_window["from"]/meta_data["duration"]*meta_data["number_samples"])
    toIndex   = int(time_window["to"]/meta_data["duration"]*meta_data["number_samples"])    
    return time[fromIndex:toIndex], signal[fromIndex:toIndex]

def getFourierTransformed(signal, meta_data):
    complexFourierTransform = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.size, d=meta_data["sampling_spacing"])    
    # nur rechte haelfte des spektrums betrachten
    complexFourierTransform = complexFourierTransform[0:int(complexFourierTransform.shape[0]/2)]
    frequencies             = frequencies[0:int(frequencies.shape[0]/2)]    
    # amplituden und phasengang bestimmen
    amplitude   = np.abs(complexFourierTransform)
    phase       = np.angle(complexFourierTransform)    
    return frequencies, amplitude, phase

def getPower(signal):
    y = signal
    M = y.shape[0]
    P = 0
    for n in range(0,M):
        P += y[n]*y[n]
    P = P*1/(M+1)
    return P

def calculateSNR(data, meta_data, time_windowNoise, time_windowSignal):
    # Lade Daten
    pNoise = list()
    pSigna = list()
    snr_FC = list()
    snr_DB = list()    
    K_values = [1.0000000, 1.12685472, 1.06518856, 1.47356999, 1.19323192, 1.54429486, 1.38300818, 1.21535447]

    for channel in [1,2,3,4,5,6,7,8]:
        # Für jedes Mikrofon die Signalleistungen Bestimmen
        signal = data[channel]          
        # Schneide Aufnahmen
        time2, signal2 = cutSignal(signal, meta_data, time_windowNoise)
        time3, signal3 = cutSignal(signal, meta_data, time_windowSignal)        
        pNoise.append(K_values[channel-1]*getSignalPower_UsingTime_AverageFree(signal2))
        pSigna.append(K_values[channel-1]*getSignalPower_UsingTime_AverageFree(signal3)-getSignalPower_UsingTime_AverageFree(signal2))
        snr_FC.append((getSignalPower_UsingTime_AverageFree(signal3)-getSignalPower_UsingTime_AverageFree(signal2))/(getSignalPower_UsingTime_AverageFree(signal2)))
        snr_DB.append(10*np.log((getSignalPower_UsingTime_AverageFree(signal3)-getSignalPower_UsingTime_AverageFree(signal2))/(getSignalPower_UsingTime_AverageFree(signal2))))    
    return pNoise, pSigna, snr_FC, snr_DB
        
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


# Load TimeFrames for Measurements    
keyList, limitList = loadTimeWindows()

# Define Array 5
micList = list()
num_mic = 8
num_mic = 8
micList.append(getPoint(-0.45-0.04,+0.90))
micList.append(getPoint(+0.45+0.04,+0.90))
micList.append(getPoint(0,+0.45))
micList.append(getPoint(-0.45-0.04,0))
micList.append(getPoint(+0.45+0.04,0))
micList.append(getPoint(0,-0.45))
micList.append(getPoint(-0.45-0.04,-0.90))
micList.append(getPoint(+0.45+0.04,-0.90))    
        
# Load Measurements
signals   = ["CH", "SX"]
distances = ["10", "20", "40", "60", "80"]
anglesA4  = ["00", "22,5", "45", "67,5", "90"]
anglesA5  = ["00", "10", "25"]

file = "results_A4_SX.txt"
f = open(file, 'w')
print("arr ; signal ; dist ; angle ; pNoise ; pSigna ; snr_DB ; K_True ; K_Estim ; Dist_Estim")
print("arr ; signal ; dist ; angle ; pNoise ; pSigna ; snr_DB ; K_True ; K_Estim ; Dist_Estim", file=f)

xval = 0.8
arr = "A4"
for signal in ["SX"]:#signals:
    for dist in distances:
        for angle in anglesA4:
            print(arr,signal,angle,dist)
            
            # Determine Sound Source Pos
            source_pos = getPoint(float(dist)*np.sin(angle_radians(float(angle.replace(",",".")))), float(dist)*np.cos(angle_radians(float(angle.replace(",",".")))))

            # Load Data
            data, meta_data = load_measurement(arr, signal, angle, dist)

            # Determine SNR
            index = keyList.index(arr+","+signal+","+dist+","+angle)
            time_windowNoise  = {"from" : limitList[index][0], "to": limitList[index][1]}
            time_windowSignal = {"from" : limitList[index][2], "to": limitList[index][3]}
            pNoise, pSigna, snr_FC, snr_DB = calculateSNR(data, meta_data, time_windowNoise, time_windowSignal)
            signalsPower = pSigna
            
            # Determine TDOA
            idxA = 8
            idxB = 2
            idxC = 2
            idxD = 4
            micA = micList[idxA-1]
            micB = micList[idxB-1]
            micC = micList[idxC-1]
            micD = micList[idxD-1]
            signalA = data[idxA]
            signalB = data[idxB]
            signalC = data[idxC]
            signalD = data[idxD]
            signalAF = butterWorthFilter(signalA, meta_data["sampling_rate"], 2000)
            signalBF = butterWorthFilter(signalB, meta_data["sampling_rate"], 2000)
            signalCF = butterWorthFilter(signalC, meta_data["sampling_rate"], 2000)
            signalDF = butterWorthFilter(signalD, meta_data["sampling_rate"], 2000)
            TDOA_CSOM1, t, csom = tdoa_csom(signalAF, signalBF, fs=meta_data["sampling_rate"], window=200)
            TDOA_CSOM2, t, csom = tdoa_csom(signalCF, signalDF, fs=meta_data["sampling_rate"], window=200)
            TDOA_real1 = getRealTDOA(source_pos, micA, micB)
            TDOA_real2 = getRealTDOA(source_pos, micC, micD)
            
            # Get Real Data
            angleReal    = 90-angle_degree(getAngle_angle1(getPoint(0,0), source_pos))
            distanceReal = distance(source_pos, getPoint(0,0))
            
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
            
            print(arr, ";", signal, ";", dist, ";", angle, ";", pNoise, ";", pSigna, ";", snr_DB, ";", K_true, ";", K_estim, ";", distanceReal, ";", distanceEstim)
            print(arr, ";", signal, ";", dist, ";", angle, ";", pNoise, ";", pSigna, ";", snr_DB, ";", K_true, ";", K_estim, ";", distanceReal, ";", distanceEstim, file=f)