#!/usr/bin/env python3
# Sorting of the loading functions into a class

import scipy.io.wavfile as wav
import numpy as np
from scipy.signal import butter, lfilter

class Loader():
    def __init__(self, sound_file_path, prefix):
        self.path = sound_file_path + prefix

        self.meta_data = {}
        self.measurements = []
        for i in range(1, 9):
            __, raw = wav.read(self.path + "-0" + str(i) + ".wav")
            arr = raw.astype(float)
            #arr = arr.reshape([np.size(arr), 1])
            self.measurements.append(arr)

        meta, raw = wav.read(self.path + "-01.wav")
        self.meta_data["sampling_rate"] = meta
        self.meta_data["number_samples"] = np.size(raw)
        self.meta_data["duration"] =  self.meta_data["number_samples"] / self.meta_data["sampling_rate"]
        self.meta_data["sampling_spacing"] = 1 / self.meta_data["sampling_rate"]
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

    #https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    def filter_butter_band_pass(self, low_cut, high_cut, order=3):
        for i in range(0, len(self.measurements)):
            x = self.measurements[i]

            nyq = 0.5 * self.meta_data["sampling_rate"]
            low = low_cut / nyq
            high = high_cut / nyq
            b, a = butter(order, [low, high], btype='band')

            self.measurements[i] = lfilter(b, a, self.measurements[i])




    #def get_time_frame(self, begin, sample_rate, end, mic_nr):
    #    
    #    return

    #def get_time_frame_fft(self, begin, end, sample_rate,)
