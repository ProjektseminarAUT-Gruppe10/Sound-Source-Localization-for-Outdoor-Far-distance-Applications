import json
import numpy as np

class SimulationLoader():
    def __init__(self, signals, sample_rate):
        self.measurements = signals
        self.meta_data = {}
        self.meta_data["sampling_rate"] = sample_rate
        self.meta_data["number_samples"] = len(signals[0])
        self.meta_data["duration"] = self.meta_data["number_samples"] / \
            self.meta_data["sampling_rate"]
        self.meta_data["sampling_spacing"] = 1 / \
            self.meta_data["sampling_rate"]
        #self.meta_data["number_streams"] = 8

    def get_measurements(self):
        return self.measurements

    def get_meta_data(self):
        return self.meta_data

    def get_ffts(self):
        ffts = []
        for m in self.measurements:
            ffts.append(np.fft.fft(m))

        return ffts

    
