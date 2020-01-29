# Imports
import sys
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

import matplotlib.pyplot as plt
import sys

import CalibrationLibrary
import GeometryLibrary
import loader



# Methods
def loadTimeWindows():
    data = genfromtxt('CutResults-A5.csv', delimiter=';',dtype="str")    
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



















def calcKs(signals, meta_data, source_pos, plot_):
    #also plot them
    if plot_:
        plt.figure()

        x = range(0, len(signals[0]))
        for i in range(0, len(signals)):
            plt.subplot(8, 1, i + 1)

            plt.plot(x, signals[i])

        plt.show()

    #main calculation
    array = GeometryLibrary.calculateMicrophoneArray_2(0.38, GeometryLibrary.getPoint(0, 0))
    distances = [] #TODO calculate them
    for i in range(0, len(signals)):
        distances.append(GeometryLibrary.distance(source_pos, array[i]))
    speed_of_sound = 343.3
    sample_rate = meta_data["sampling_rate"]
    calib = CalibrationLibrary.Calibrator(signals, distances, speed_of_sound, sample_rate)
    ks = calib.run_calibration()
    return ks


# Load TimeFrames for Measurements    
keyList, limitList = loadTimeWindows()
          
# Load Measurements
signals   = ["CH", "SX"]
distances = ["10", "20", "40", "60", "80"]
anglesA4  = ["00", "22,5", "45", "67,5", "90"]
anglesA5  = ["00", "10", "25"]

f = open("CalibrationResults.txt", 'w')
print("arr ; signal ; angle ; dist ; K1 ; K2 ; K3 ; K4 ; K5 ; K6 ; K7 ; K8")
print("arr ; signal ; angle ; dist ; K1 ; K2 ; K3 ; K4 ; K5 ; K6 ; K7 ; K8", file=f)

arr = "A5"
for signal in signals:
    for dist in distances:
        for angle in anglesA5:            
            # Determine Sound Source Pos
            source_pos = getPoint(float(dist)*np.sin(angle_radians(float(angle.replace(",",".")))), float(dist)*np.cos(angle_radians(float(angle.replace(",",".")))))
            
            # Load Data
            data, meta_data = load_measurement(arr, signal, angle, dist)

            # Determine SNR
            index = keyList.index(arr+","+signal+","+dist+","+angle)
            time_windowNoise  = {"from" : limitList[index][0], "to": limitList[index][1]}
            time_windowSignal = {"from" : limitList[index][2], "to": limitList[index][3]}

            # Determine Ks
            signalsCut = list()
            for d in range(1,9):
                timeCut, sigCut = cutSignal(data[d], meta_data, time_windowSignal)   
                signalsCut.append(sigCut)
            K_results = calcKs(signalsCut, meta_data, source_pos, plot_=False)
#            print(K_results)
            print(arr,";",signal,";",angle,";",dist,";",K_results[0],";",K_results[1],";",K_results[2],";",K_results[3],";",K_results[4],";",K_results[5],";",K_results[6],";",K_results[7])
            print(arr,";",signal,";",angle,";",dist,";",K_results[0],";",K_results[1],";",K_results[2],";",K_results[3],";",K_results[4],";",K_results[5],";",K_results[6],";",K_results[7], file=f)
f.close()