import os, sys
import numpy as np
import math

power_path = os.path.join('..', 'libraries')
sys.path.append(power_path)

import SignalProcessingLibrary

class Calibrator():
    def __init__(self, signals, distances_to_source, speed_of_sound, sample_rate):
        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.signals = signals
        self.distances_to_source = distances_to_source
        self.dt = len(self.signals[0])
        self.delays = []
        self.ks = []
        self.powers = []
        self.amplitudes = []

    def calculate_delays(self):
        for d in self.distances_to_source:
            delay = d / self.speed_of_sound * self.sample_rate
            delay = int(round(delay))
            self.delays.append(delay)


    def get_usable_signals(self):
        max_delay = max(self.delays)

        self.dt = self.dt - max_delay  # adjust length to useable part

        # cut signals to useable part
        for i in range(0, len(self.signals)):
            
            self.signals[i] = self.signals[i][self.delays[i]:self.delays[i] + self.dt]

    def calculate_powers(self):
        for s in self.signals:
            
            power = SignalProcessingLibrary.getSignalPower_UsingTime_AverageFree(np.asarray(s))
            #power = SignalProcessingLibrary.getSignalPower_UsingTime(np.asarray(s))
            #power = np.sum(np.asarray(s)) / (len(s) + 1)
            #print(power)
            self.powers.append(power)

    def calculate_amplitudes(self):
        for s in self.signals:
            amplitude = max(s)
            #amplitude = sum(s)/len(s)
            self.amplitudes.append(amplitude)

    def calibrate(self):
        self.ks.append(1.0)

        # now calibrate microphones
        for i in range(1, len(self.signals)):
            self.ks.append(1.0 * pow(self.distances_to_source[0], 2) * self.ks[0] * self.powers[0] * 1/pow(self.distances_to_source[i], 2) / self.powers[i])

        #trying to set ksource to 1.0
        #for i in range(0, len(self.signals)):
        #    self.ks.append(1.0 / pow(self.distances_to_source[i], 2) / self.powers[i])

    def calibrate_amplitudes(self):
        self.ks.append(1.0)

        for i in range(1, len(self.signals)):
            self.ks.append(1.0 * self.distances_to_source[0] * self.ks[0] * self.amplitudes[0] / self.distances_to_source[i] / self.amplitudes[i])

    def run_calibration(self):
        self.calculate_delays()
        self.get_usable_signals()
        self.calculate_powers()
        self.calibrate()

        return self.ks

    def run_calibration_amplitude(self):
        self.calculate_delays()
        self.get_usable_signals()
        self.calculate_amplitudes()
        self.calibrate_amplitudes()

        return self.ks

