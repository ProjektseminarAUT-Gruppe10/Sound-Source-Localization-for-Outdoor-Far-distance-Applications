# -*- coding: utf-8 -*-

from scipy.io.wavfile import read
import numpy as np
import matplotlib.pyplot as plt

def load_measurement(array, soundSource, angle, distance):    
    meta_data = {}
    data = read("../../measurements/measurement_3/"+array+"_"+soundSource+"_"+angle+"_"+distance+"-01.wav")
    meta_data["sampling_rate"]    = data[0]                         # Sampling Rate in Hz
    meta_data["duration"]         = data[1].shape[0]/data[0]        # Duration in seconds
    meta_data["sampling_spacing"] = 1/meta_data["sampling_rate"]    # Time distance between two samples
    meta_data["number_samples"]   = data[1].shape[0]                # Total number of samples
    
    data = np.array(data[1],dtype=float)
    data = np.reshape(data,(data.shape[0],1))
    for i in [1]:#,2,3,4,5,6,7,8):
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

def calculateSNR(arr, sig, dis, angl):
    # Lade Daten
    data, meta_data = load_measurement(arr, sig, angl, dis)
    
    # Für jedes Mikrofon die SNR bestimmen
    channel = 1
    signal = data[channel]  
    time = np.arange(meta_data["number_samples"])/meta_data["sampling_rate"] 
    plt.title(arr+" "+sig+" "+dis+" "+angl)
    plt.plot(time,signal)
    plt.show()
    plt.pause(10)
    plt.clf()
    
# Calculate all SNR Levels
arrays = ["A4", "A5"]
signals = ["CH", "SX"]
distances = ["10", "20", "40", "60", "80"]
anglesA4 = ["00", "22,5", "45", "67,5", "90"]
anglesA5 = ["00", "10", "25"]

print("Array","\t","Signal","\t","Distance","\t","Angle")
for arr in arrays:
    for sig in signals:
        for dis in distances:
            z = anglesA4
            if(arr=="A5"):
                z = anglesA5
            for angl in z:
                print(arr,"\t",sig,"\t",dis,"\t",angl)
                index = keyList.index(arr+","+sig+","+dis+","+angl)
                time_windowNoise  = {"from" : limitList[index][0], "to": limitList[index][1]}
                time_windowSignal = {"from" : limitList[index][2], "to": limitList[index][3]}
                calculateSNR(arr, sig, dis, angl, time_windowNoise, time_windowSignal)