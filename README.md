## Projektseminar AUT - WS2019/20 - TU Darmstadt 
# "Sound Source Localization for Outdoor, Far-distance Applications"
Students:   Frederik Alexander Bark, Kevin Riehl, Leonardo Zaninelli
Supervisor: M.Sc. Yuri Furletov
### Introduction
This is the github repository of our project seminar containing all code, libraries,
measurement data, experiments and results. The purpose of this work was to explore ways
to improve distance estimation accuracy for sound source localization purposes within autonomous
driving applications. 

### Table of contents
- [Introduction](#Introduction)
- [Table of contents](#Table-of-contents)
- [Structure of this repository](#Structure-of-this-repository)
- [Simple example how to use](#Simple-example-how-to-use)

### Structure of this repository
> *ProjektseminarAUT/measurements*

This folder contains four measurements and uses following naming convention for the files:
**ARRAY_SOUNDSOURCE_ANGLE_DISTANCE-CHANNEL.wav**
(array, sound source, angle, distance and channel each are denominated by two digits)
Moreover, each folder contains a PDF file with a detailed outline how measurements were performed.
*Note:* "measurement_0" was recorded and provided by supervisor prior to the seminar.

> *ProjektseminarAUT/experiments*

This folder contains all code and related results in TXT and Excel. Note, experiments 4 and 8 
were repeated due to mistakes. The experiments furthermore deliver a good example for how to
use the complex library and parts of the SSL stacks for research purposes.

> *ProjektseminarAUT/latex*

This folder contains all files related to the paper written by the team for this seminar.

> *ProjektseminarAUT/libraries*

This folder contains all libraries developed for the purpose of implementing the SSL stack
mentioned in our paper. A list of all libraries follows:
- AnimationLibrary
- CalibrationLibrary
- GeometryLibrary
- GraphicLibrary
- MicrophonePositionLibrary
- OptimizationLibrary
- SignalPRocessingLibrary
- SimulationLibrary
- SslLibrary
- TDOA_Library

Each library contains a set of methods with explanations about parameters, return values
and purpose on its head. Moreover, at the end of each file, there are some small demos (commented out)
that can be used to orientate and to understand how several library methods can be used.

Moreover, several folders such as *ProjektseminarAUT/calibration*, *ProjektseminarAUT/simulation*
and *ProjektseminarAUT/tdoa* exist, that contain codes that are called by the python scripts within
the libraries folder.

### Simple Example how to use
First of all, microphone positions have to be defined, either manually...
```
from GeometryLibrary import getPoint

micA = getPoint(-1,0)
micB = getPoint(+1,0)
#...
micList = list()
micList.append(micA)
micList.append(micB)
#...
```
or by defining an array that was used (e.g. circular array)...
```
from MicrophonePositionLibrary import getMicrophonePositions_SIM_A
micList = getMicrophonePositions_SIM_A(radius=0.5)
```

Second, auditory data has to be gathered, either by loading them from files...
```
import numpy as np
from scipy.io.wavfile import read

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
    
signals, meta = load_measurement("A4", "CH", "22,5", "40")
```
or by simulating them. For simulations, several parameters have to be defined and adjusted within a "configEXP.json" 
file (e.g. the noise parameters, for an example see "experiments/k_experiment1/configEXP.json"). In addition to that,
a signal function for the source has to be defined...
```
import math
from SimulationLibrary import load_configs, simulate

# Signal Function
def signal_function(x): 
    return 0.58 * math.sin(x * (2 * math.pi * 400.0)) if (x > 0.05 and x < 0.1) else 0

# Load Configuration
config = load_configs("configEXP.json")[0]

# Set microphone and sound source position, if not declared within json file
#config["microphone_noise_mus"] = list()
#config["microphone_noise_sigmas"] = list()
#config["microphone_noise_amplitudes"] = list()
#config["microphone_positions"] = list()
#
#config["microphone_noise_mus"].append(0.01)
#config["microphone_noise_sigmas"].append(0.02)
#config["microphone_noise_amplitudes"].append(1.0)
#config["microphone_positions"].append(getPoint(-1, 0))
#
#config["microphone_noise_mus"].append(0.01)
#config["microphone_noise_sigmas"].append(0.02)
#config["microphone_noise_amplitudes"].append(1.0)
#config["microphone_positions"].append(getPoint(+1, 0))
#
#config["source_position"] = getPoint(5, 10)
#config["source_noise_sigma"] = 0.2
#config["general_noise_sigma"] = 0.3

# Simulation
simData = simulate(config, config["source_position"], signal_function)
signals = loaded.get_measurements()
meta    = loaded.get_meta_data()
signalA = signals[0]
signalB = signals[1]
#...
```

Consecutively, the auditory data signals can be filtered...
```
from SignalProcessingLibrary import butterWorthFilter, wienerFilter

signalAF = wienerFilter(signalA)
signalBF = butterWorthFilter(signalB, meta["sampling_rate"], 1500)
```

Afterwards several features such as TDOA or power can be extracted from the data:
TDOA estimation:
```
from SignalProcessingLibrary import tdoa_csom, tdoa_gcc_phat

TDOA_CSOM, t, csom = tdoa_csom(signalAF, signalBF, fs=meta["sampling_rate"], window=500)
TDOA_GCCP, cc = tdoa_gcc_phat(signalAF, signalBF, fs=meta["sampling_rate"])
```
Signal Power estimation:
```
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree, tdoa_csom

powerSigA = getSignalPower_UsingTime_AverageFree(signalAF)
powerSigB = getSignalPower_UsingTime_AverageFree(signalBF)

# Calculate K estimation
K_estim = list()
for i in range(0,len(micList)):
    for j in range(0, len(micList)):
        if(i!=j):
            TDOA_CSOM, t, csom = tdoa_csom(signalF[i], signalF[j], fs=meta["sampling_rate"], window=2000)
            K1_estim, K2_estim = estimateK_Pair(signalsPower[i], signalsPower[j], micList[i], micList[j], TDOA_CSOM*343.2)
            K_estim.append(K1_estim)
            K_estim.append(K2_estim)
            
# Remove NAN
K_estim = [x for x in K_estim if str(x) != 'nan']
K_estim.sort()            
xval = 0.80
K_val_estim = K_estim_noise[int(len(K_estim)*xval)]

# Distance Estimated
radiusList = list()
for i in range(0,len(micList)):
    radiusList.append(np.sqrt(K_val_estim/signalsPower[i]))
distanceEstim = np.average(radiusList)
```

Once this is done, SSL can be performed using three different approaches mentioned in the PAPER:
```
from SslLibrary import SSL_TDOA_LN, SSL_TDOA_NL, SSL_AMPL

pointsLIN = SSL_TDOA_LN(micPosList, tdoa1, tdoa2, c_speed=343.2)
pointsNOL = SSL_TDOA_NL(micPosList, tdoa1, tdoa2, c_speed=342.2, LIN=pointsLIN[0])
pointsAMP = SSL_AMPL(micPosList, distA, distB)
```
