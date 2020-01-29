# -*- coding: utf-8 -*-

from numpy import genfromtxt
from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("..\\..\\libraries")
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree

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

def load_measurement(array, soundSource, angle, distance):    
    meta_data = {}
    data = read("../../measurements/measurement_3/"+array+"_"+soundSource+"_"+angle+"_"+distance+"-01.wav")
    meta_data["sampling_rate"]    = data[0]                         # Sampling Rate in Hz
    meta_data["duration"]         = data[1].shape[0]/data[0]        # Duration in seconds
    meta_data["sampling_spacing"] = 1/meta_data["sampling_rate"]    # Time distance between two samples
    meta_data["number_samples"]   = data[1].shape[0]                # Total number of samples
    
    data = np.array(data[1],dtype=float)
    data = np.reshape(data,(data.shape[0],1))
    for i in (1,2,3,4,5,6,7,8):
        b = read("../../measurements/measurement_3/"+array+"_"+soundSource+"_"+angle+"_"+distance+"-0"+str(i)+".wav")
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

def calculateSNR(arr, sig, dis, angl, time_windowNoise, time_windowSignal):
    # Lade Daten
    data, meta_data = load_measurement(arr, sig, angl, dis)
    
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

# Load TimeFrames for Measurements    
keyList, limitList = loadTimeWindows()

# Calculate all SNR Levels
arrays = ["A4", "A5"]
signals = ["CH", "SX"]
distances = ["10", "20", "40", "60", "80"]
anglesA4 = ["00", "22,5", "45", "67,5", "90"]
anglesA5 = ["00", "10", "25"]

f = open('out.txt', 'w')
print("Array","\t","Signal","\t","Distance","\t","Angle","\t","Channel","\t","Pnoise[W]","\t","Psignal[W]","\t","SNR_FAC","\t","SNR_DB[db]", file=f)  # Python 3.x
print("Array","\t","Signal","\t","Distance","\t","Angle","\t","Channel","\t","Pnoise[W]","\t","Psignal[W]","\t","SNR_FAC","\t","SNR_DB[db]")
for arr in arrays:
    for sig in signals:
        for dis in distances:
            z = anglesA4
            if(arr=="A5"):
                z = anglesA5
            for angl in z:
                index = keyList.index(arr+","+sig+","+dis+","+angl)
                time_windowNoise  = {"from" : limitList[index][0], "to": limitList[index][1]}
                time_windowSignal = {"from" : limitList[index][2], "to": limitList[index][3]}
                pNoise, pSigna, snr_FC, snr_DB = calculateSNR(arr, sig, dis, angl, time_windowNoise, time_windowSignal)
                              
                print(arr,"\t",sig,"\t",dis,"\t",angl,"\t","average","\t",np.average(pNoise),"\t",np.average(pSigna),"\t",np.average(snr_FC),"\t",np.average(snr_DB), file=f)
                print(arr,"\t",sig,"\t",dis,"\t",angl,"\t","average","\t",np.average(pNoise),"\t",np.average(pSigna),"\t",np.average(snr_FC),"\t",np.average(snr_DB))
                for c in range(0,7):
                    print(arr,"\t",sig,"\t",dis,"\t",angl,"\t",str(c+1),"\t",pNoise[c],"\t",pSigna[c],"\t",snr_FC[c],"\t",snr_DB[c], file=f)
                    print(arr,"\t",sig,"\t",dis,"\t",angl,"\t",str(c+1),"\t",pNoise[c],"\t",pSigna[c],"\t",snr_FC[c],"\t",snr_DB[c])
f.close()