# Import
import numpy as np
from scipy import signal as sg

# Signal Processing Library
def getFourierTransformed(signal, sampling):
    complexFourierTransform = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal.size, 1/sampling)
    # nur rechte haelfte des spektrums betrachten
    complexFourierTransform = complexFourierTransform[0:int(complexFourierTransform.shape[0]/2)]
    frequencies             = frequencies[0:int(frequencies.shape[0]/2)]
    # amplituden und phasengang bestimmen
    amplitude   = np.abs(complexFourierTransform)
    phase       = np.angle(complexFourierTransform)    
    return frequencies, amplitude, phase

def getSignalPower_UsingTime(t_signal):
    return np.power(np.linalg.norm(t_signal),2)/((t_signal.shape[0]+1))

def getSignalPower_UsingTime_AverageFree(t_signal):
    t_signal -= np.average(t_signal)
    return getSignalPower_UsingTime(t_signal)

def centralSecondOrderMomentFunction(signal1, signal2):
    signalA = signal1 - np.average(signal1)
    signalB = signal2 - np.average(signal2)
    return secondOrderMomentFunction(signalA, signalB)
    
def secondOrderMomentFunction(signal1, signal2):
    N = signal1.shape[0]
    a = np.zeros((N-1))
    for m in range(0,(N-1)):
        counter = 0
        for n in range(0,N-m-1):
            counter += 1
            a[m] += signal1[n+m]*signal2[n]  #https://www.mathworks.com/help/matlab/ref/xcorr.html#mw_ff426c84-793b-4341-86f0-077eba46eb22
        a[m] = 1/(counter+1) * a[m]
    b = np.flip(np.delete(a,0))
    c = np.concatenate((b,a))
    t = np.zeros_like(c)
    for i in range(0,t.shape[0]):
        t[i] = i-b.shape[0] 
    return t, c

def centralSecondOrderMomentFunction_part(signal1, signal2, window):
    signalA = signal1 - np.average(signal1)
    signalB = signal2 - np.average(signal2)
    return secondOrderMomentFunction_part(signalA, signalB, window)
    
def secondOrderMomentFunction_part(signal1, signal2, window):
    N = signal1.shape[0]
    a = np.zeros((N-1))
    for m in range(0,window):
        counter = 0
        for n in range(0,N-m-1):
            counter += 1
            a[m] += signal1[n+m]*signal2[n]  #https://www.mathworks.com/help/matlab/ref/xcorr.html#mw_ff426c84-793b-4341-86f0-077eba46eb22
        a[m] = 1/(counter+1) * a[m]
    
    b = np.zeros((N-1))
    for m in range(0,window):
        counter = 0
        for n in range(0,N-m-1):
            counter += 1
            b[m] += signal1[n]*signal2[n+m]  #https://www.mathworks.com/help/matlab/ref/xcorr.html#mw_ff426c84-793b-4341-86f0-077eba46eb22
        b[m] = 1/(counter+1) * b[m]    
    
    a = np.delete(a,0)
#    b = np.flip(np.delete(a,0))
    b = np.flip(b)
    c = np.concatenate((b,a))
    t = np.zeros_like(c)
    for i in range(0,t.shape[0]):
        t[i] = i-b.shape[0]+1
    return t, c

def getPeriodogramm_Estimation_4_CXX(signal, sampling):
    f_period, Pxx_den_period = sg.periodogram(signal, sampling, scaling='spectrum')
    return f_period, Pxx_den_period

def getWelch_Estimation_4_CXX(signal, sampling):
    f_welch, Pxx_den_welch = sg.welch(signal, sampling, nperseg=1024)
    return f_welch, Pxx_den_welch
    
def wienerFilter(signal):
    return sg.wiener(signal)
    
def butterWorthFilter(signal, sampling_freq, cutOffFreq):
    fc = cutOffFreq  # Cut-off frequency of the filter
    w = fc / (sampling_freq / 2) # Normalize the frequency
    b, a = sg.butter(5, w, 'low')
    filtered = sg.filtfilt(b, a, signal)    
    return filtered

def tdoa_gcc_phat(sig, refsig, fs=1, max_tau=None, interp=1):
    '''
    This function computes the offset between the signal sig and the reference signal refsig
    using the Generalized Cross Correlation - Phase Transform (GCC-PHAT)method.
    '''
    sig = np.asarray(sig)
    refsig = np.asarray(refsig)
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

def tdoa_csom(sig, refsig, fs=1, window=500):
    t, cSOM = centralSecondOrderMomentFunction_part(sig, refsig, window)
    idx = np.argmax(cSOM)
    shift = t[idx]
    TDOA =  shift / float(fs) 
    return TDOA, t, cSOM