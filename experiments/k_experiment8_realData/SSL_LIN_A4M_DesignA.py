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
    for channel in [1,2,3,4,5,6,7,8]:
        # Für jedes Mikrofon die Signalleistungen Bestimmen
        signal = data[channel]          
        # Schneide Aufnahmen
        time2, signal2 = cutSignal(signal, meta_data, time_windowNoise)
        time3, signal3 = cutSignal(signal, meta_data, time_windowSignal)        
        pNoise.append(getSignalPower_UsingTime_AverageFree(signal2))
        pSigna.append(getSignalPower_UsingTime_AverageFree(signal3)-getSignalPower_UsingTime_AverageFree(signal2))
        snr_FC.append((getSignalPower_UsingTime_AverageFree(signal3)-getSignalPower_UsingTime_AverageFree(signal2))/(getSignalPower_UsingTime_AverageFree(signal2)))
        snr_DB.append(10*np.log((getSignalPower_UsingTime_AverageFree(signal3)-getSignalPower_UsingTime_AverageFree(signal2))/(getSignalPower_UsingTime_AverageFree(signal2))))    
    return pNoise, pSigna, snr_FC, snr_DB

def SSL_TDOA_LIN(tdoa1, tdoa2, micA, micB, micC, micD):        
    std1 = tdoa1*343.2
    std2 = tdoa2*343.2
    steep1 = KarstenDOA_calculateSteep_linear_simple(distance(micA, micB), std1)
    steep2 = KarstenDOA_calculateSteep_linear_simple(distance(micC, micD), std2)
    solutions, m1_a, m1_b, m2_a, m2_b, b1_a, b1_b, b2_a, b2_b = getMicrophonePair_DOA_Intersection_linear(micA, micB, micC, micD, steep1, steep2)
    estimation = plausibleFilter_TDOA(solutions)
    return estimation

def SSL_TDOA_NOL(tdoa1, tdoa2, micA, micB, micC, micD, estimationLIN):    
    std1 = tdoa1*343.2
    std2 = tdoa2*343.2
    curveA = get_tCurve(micA, micB, std1)
    curveB = get_tCurve(micC, micD, std2)
    estimationNOL = optimizeIntersectionPoint_nonLinear_numeric(estimationLIN, curveA, curveB)
    return estimationNOL

def plausibleFilter_TDOA(solutions):
#    print(solutions)
    result = list()
    for p in solutions:
        if(p[1]>=0 and p[0]>=0):
            result.append(p)
    if(len(result)==1):
        return result[0]
    elif(len(result)==2):
        if(np.linalg.norm(result[0])>np.linalg.norm(result[1])):
            return result[0]
        else:
            return result[1]    
    else:
        return "no"
        
def getRealDeltaS(source_pos, micAPos, micBPos):
    return distance(micAPos, source_pos) - distance(micBPos, source_pos)

def getRealTDOA(source_pos, micAPos, micBPos):
    return getRealDeltaS(source_pos, micAPos, micBPos) / 343.3


# Load TimeFrames for Measurements    
keyList, limitList = loadTimeWindows()

# Define Array 5
micList = list()
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

file = "results_SSL_LIN_A4M_DesignA.txt"
f = open(file, 'w')
print("arr ; signal ; angle ; dist ; pNoise ; pSigna ; snr_DB ; TDOA_real1 ; TDOA_real2 ; TDOA_CSOM1 ; TDOA_CSOM2 ; angleReal ; angleLIN ; angleNOL ; distanceReal ; distanceLIN ; distanceNOL")
print("arr ; signal ; angle ; dist ; pNoise ; pSigna ; snr_DB ; TDOA_real1 ; TDOA_real2 ; TDOA_CSOM1 ; TDOA_CSOM2 ; angleReal ; angleLIN ; angleNOL ; distanceReal ; distanceLIN ; distanceNOL", file=f)

arr = "A4"
for signal in signals:
    for dist in distances:
        for angle in anglesA4:
#            signal = "CH"
#            angle  = "25"
#            dist   = "10"
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
            # Determine TDOA
            idxA = 1
            idxB = 4
            idxC = 4
            idxD = 7
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
            
            # Determine SSL LIN
            estimationLIN = SSL_TDOA_LIN(TDOA_CSOM1, TDOA_CSOM2, micA, micB, micC, micD)            
            if(estimationLIN=="no"):                
                print(arr, ";", signal, ";", angle, ";", dist, ";", pNoise, ";", pSigna, ";", snr_DB, ";", TDOA_real1, ";", TDOA_real2, ";", TDOA_CSOM1, ";", TDOA_CSOM2, ";", angleReal, ";", "nan", ";", "nan", ";", distanceReal, ";", "nan", ";", "nan")
                print(arr, ";", signal, ";", angle, ";", dist, ";", pNoise, ";", pSigna, ";", snr_DB, ";", TDOA_real1, ";", TDOA_real2, ";", TDOA_CSOM1, ";", TDOA_CSOM2, ";", angleReal, ";", "nan", ";", "nan", ";", distanceReal, ";", "nan", ";", "nan", file=f)
                continue;
                
            # Determine SSL NOL
            estimationNOL = SSL_TDOA_NOL(TDOA_CSOM1, TDOA_CSOM2, micA, micB, micC, micD, estimationLIN)
            
            # Fehler Auswertung
            angleLIN    = 90-angle_degree(getAngle_angle1(getPoint(0,0), estimationLIN))
            angleNOL    = 90-angle_degree(getAngle_angle1(getPoint(0,0), estimationNOL))            
            distanceLIN = distance(source_pos, estimationLIN)
            distanceNOL = distance(source_pos, estimationNOL)
            
            TDOA_error1  = (TDOA_real1 - TDOA_CSOM1)
            TDOA_error2  = (TDOA_real2 - TDOA_CSOM2)
            distErrorLIN = (distanceReal - distanceLIN)
            distErrorNOL = (distanceReal - distanceNOL)
            anglErrorLIN = (angleReal - angleLIN)
            anglErrorNOL = (angleReal - angleNOL)
            
            print(angleLIN, angleNOL)
            print(distanceLIN, distanceNOL)
            
            print(arr, ";", signal, ";", angle, ";", dist, ";", pNoise, ";", pSigna, ";", snr_DB, ";", TDOA_real1, ";", TDOA_real2, ";", TDOA_CSOM1, ";", TDOA_CSOM2, ";", angleReal, ";", angleLIN, ";", angleNOL, ";", distanceReal, ";", distanceLIN, ";", distanceNOL)
            print(arr, ";", signal, ";", angle, ";", dist, ";", pNoise, ";", pSigna, ";", snr_DB, ";", TDOA_real1, ";", TDOA_real2, ";", TDOA_CSOM1, ";", TDOA_CSOM2, ";", angleReal, ";", angleLIN, ";", angleNOL, ";", distanceReal, ";", distanceLIN, ";", distanceNOL, file=f)
