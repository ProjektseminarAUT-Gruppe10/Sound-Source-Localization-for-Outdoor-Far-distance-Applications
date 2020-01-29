# Imports
import numpy as np
import SignalProcessingLibrary

class Calibrator():
    def __init__(self, signals, distances_to_source, speed_of_sound, sample_rate):
        self.speed_of_sound = speed_of_sound
        self.sample_rate = sample_rate
        self.signals = signals
        self.distances_to_source = distances_to_source
        self.ks = []
        self.powers = []

    def calculate_powers(self):
        for s in self.signals: 
            power = SignalProcessingLibrary.getSignalPower_UsingTime_AverageFree(np.asarray(s))
            self.powers.append(power)

    def calibrate(self):
        self.ks.append(1.0)
        # now calibrate microphones
        for i in range(1, len(self.signals)):
            self.ks.append(1.0 * pow(self.distances_to_source[0], 2) * self.ks[0] * self.powers[0] * 1/pow(self.distances_to_source[i], 2) / self.powers[i])

    def run_calibration(self):
        self.calculate_powers()
        self.calibrate()
        return self.ks
