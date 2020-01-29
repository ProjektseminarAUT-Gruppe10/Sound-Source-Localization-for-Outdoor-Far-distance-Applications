# Dieses Skript versucht die TDOA im Zeitbereich mit CSOM zu bestimmen
# Variation der Mikrofonabstände sowie Sampling Frequenz 96kHz
# Array C: Mikrofone nebeneinander vertikal

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

def plausibleFilter_TDOA(solutions):
#    print(solutions)
    result = list()
    for p in solutions:
        if(p[1]>=0):# and p[0]>=0):
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

def calcSTDOA(signalA, signalB, micPosList, indA, indB):        
    std1 = tdoa_gcc_phat(signalA, signalB, fs=meta["sampling_rate"])[0]*343.3
    rstd1 = getRealDeltaS(source_pos, micPosList[indA], micPosList[indB])
    print("STDOA Estimated / Real: ",std1,rstd1)
   
def getRealDeltaS(source_pos, micAPos, micBPos):
    return distance(micAPos, source_pos) - distance(micBPos, source_pos)

def getRealTDOA(source_pos, micAPos, micBPos):
    return getRealDeltaS(source_pos, micAPos, micBPos) / 343.3

def updateConfig(config, micA, micB, micC, micD, microphone_noise_sigm, noise_environment, noise_source, source_pos):
    config["microphone_noise_mus"] = list()
    config["microphone_noise_sigmas"] = list()
    config["microphone_noise_amplitudes"] = list()
    config["microphone_positions"] = list()
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_mus"].append(0)
    config["microphone_noise_sigmas"].append(microphone_noise_sigm)
    config["microphone_noise_sigmas"].append(microphone_noise_sigm)
    config["microphone_noise_sigmas"].append(microphone_noise_sigm)
    config["microphone_noise_sigmas"].append(microphone_noise_sigm)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_noise_amplitudes"].append(1)
    config["microphone_positions"].append(micA)
    config["microphone_positions"].append(micB)
    config["microphone_positions"].append(micC)
    config["microphone_positions"].append(micD)
    config["source_position"] = source_pos
    config["source_noise_sigma"] = noise_source
    config["general_noise_sigma"] = noise_environment
    return config

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
N = 10

distances = [80]#[5,10,20,40,60,80,100]#[1,5,10,20,40,60,80,100]
angles    = [45, 60, 70, 80, 90]#[0, 10, 20, 30, 45, 60, 70, 80, 90]

f = open('out-dist-0,40m_Cc2.txt', 'w')
print("dis;ang;source_pos;TDOA_Fehler1;TDOA_Fehler1_std;TDOA_Fehler2;TDOA_Fehler2_std;DistanceErrorLIN;DistanceErrorLIN_std;DistanceErrorNOL;DistanceErrorNOL_std;AngleErrorLIN;AngleErrorLIN_std;AngleErrorNOL;AngleErrorNOL_std;")
print("dis;ang;source_pos;TDOA_Fehler1;TDOA_Fehler1_std;TDOA_Fehler2;TDOA_Fehler2_std;DistanceErrorLIN;DistanceErrorLIN_std;DistanceErrorNOL;DistanceErrorNOL_std;AngleErrorLIN;AngleErrorLIN_std;AngleErrorNOL;AngleErrorNOL_std;", file=f)

mic_dist = 0.4
            
for dis in distances:
    for ang in angles:
        TDOA_realls1 = list()
        TDOA_realls2 = list()
        TDOA_csomls1 = list()
        TDOA_csomls2 = list()
        TDOAstdErr1  = list()
        TDOAstdErr2  = list()
        ANGLE_stdErrLIN = list()
        DISTA_stdErrLIN = list()
        ANGLE_stdErrNOL = list()
        DISTA_stdErrNOL = list()
        
        for n in range(0,N):
            # Adjust Configurations
            micA = getPoint(-mic_dist,-mic_dist)
            micB = getPoint(-mic_dist,+mic_dist)
            micC = getPoint(+mic_dist,-mic_dist)
            micD = getPoint(+mic_dist,+mic_dist)
            noise_microphone = 0.003
            noise_environment = 0.04
            noise_source = 0.01
            source_pos = getPoint(dis*np.sin(angle_radians(ang)),dis*np.cos(angle_radians(ang)))
            config = updateConfig(config, micA, micB, micC, micD, noise_microphone, noise_environment, noise_source, source_pos)
            
            ## Starte Simulation
            loaded = simulate(config, config["source_position"], signal_function)
            signals = loaded.get_measurements()
            meta    = loaded.get_meta_data()
            signalA = signals[0]
            signalB = signals[1]
            signalC = signals[2]
            signalD = signals[3]
            signalAF = butterWorthFilter(signalA, meta["sampling_rate"], 2000)
            signalBF = butterWorthFilter(signalB, meta["sampling_rate"], 2000)
            signalCF = butterWorthFilter(signalC, meta["sampling_rate"], 2000)
            signalDF = butterWorthFilter(signalD, meta["sampling_rate"], 2000)
            
            TDOA_CSOM1, t, csom = tdoa_csom(signalAF, signalBF, fs=meta["sampling_rate"], window=2000)
            TDOA_CSOM2, t, csom = tdoa_csom(signalCF, signalDF, fs=meta["sampling_rate"], window=2000)
            TDOA_real1 = getRealTDOA(source_pos, micA, micB)
            TDOA_real2 = getRealTDOA(source_pos, micC, micD)
            
            # SSL - TDOA_LIN
            estimationLIN = SSL_TDOA_LIN(TDOA_CSOM1, TDOA_CSOM2, micA, micB, micC, micD)
            if(estimationLIN=="no"):
                continue;
            
            # SSL - TDOA_NOL
            estimationNOL = SSL_TDOA_NOL(TDOA_CSOM1, TDOA_CSOM2, micA, micB, micC, micD, estimationLIN)
            
            # Distance
            distanceReal = distance(source_pos, getPoint(0,0))
            angleReal    = 90-angle_degree(getAngle_angle1(getPoint(0,0), source_pos))
            
            distanceLIN = distance(source_pos, estimationLIN)
            distanceNOL = distance(source_pos, estimationNOL)
            angleLIN    = 90-angle_degree(getAngle_angle1(getPoint(0,0), estimationLIN))
            angleNOL    = 90-angle_degree(getAngle_angle1(getPoint(0,0), estimationNOL))
            
#            print("angles ° ",angleReal,angleLIN,angleNOL)
#            print("distances m ",distanceReal,distanceLIN, distanceNOL)

            TDOA_error1  = np.sqrt((TDOA_real1 - TDOA_CSOM1)*(TDOA_real1 - TDOA_CSOM1))
            TDOA_error2  = np.sqrt((TDOA_real2 - TDOA_CSOM2)*(TDOA_real2 - TDOA_CSOM2))
            distErrorLIN = np.sqrt((distanceReal - distanceLIN)*(distanceReal - distanceLIN))
            distErrorNOL = np.sqrt((distanceReal - distanceNOL)*(distanceReal - distanceNOL))
            anglErrorLIN = np.sqrt((angleReal - angleLIN)*(angleReal - angleLIN))
            anglErrorNOL = np.sqrt((angleReal - angleNOL)*(angleReal - angleNOL))
            
            TDOAstdErr1.append(TDOA_error1)
            TDOAstdErr2.append(TDOA_error2)
            DISTA_stdErrLIN.append(distErrorLIN)
            DISTA_stdErrNOL.append(distErrorNOL)
            ANGLE_stdErrLIN.append(anglErrorLIN)
            ANGLE_stdErrNOL.append(anglErrorNOL)
            
        print(dis, ";", ang, ";", source_pos, ";", np.average(TDOAstdErr1), ";", np.std(TDOAstdErr1), ";", np.average(TDOAstdErr2), ";", np.std(TDOAstdErr2), ";", np.average(DISTA_stdErrLIN), ";", np.std(DISTA_stdErrLIN), ";", np.average(DISTA_stdErrNOL), ";", np.std(DISTA_stdErrNOL), ";", np.average(ANGLE_stdErrLIN), ";", np.std(ANGLE_stdErrLIN), ";", np.average(ANGLE_stdErrNOL), ";", np.std(ANGLE_stdErrNOL))
        print(dis, ";", ang, ";", source_pos, ";", np.average(TDOAstdErr1), ";", np.std(TDOAstdErr1), ";", np.average(TDOAstdErr2), ";", np.std(TDOAstdErr2), ";", np.average(DISTA_stdErrLIN), ";", np.std(DISTA_stdErrLIN), ";", np.average(DISTA_stdErrNOL), ";", np.std(DISTA_stdErrNOL), ";", np.average(ANGLE_stdErrLIN), ";", np.std(ANGLE_stdErrLIN), ";", np.average(ANGLE_stdErrNOL), ";", np.std(ANGLE_stdErrNOL), file=f)
    
f.close()