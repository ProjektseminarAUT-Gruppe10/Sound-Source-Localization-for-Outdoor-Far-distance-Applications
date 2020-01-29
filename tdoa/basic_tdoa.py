#!/usr/bin/python3
import loader
import math
import array_parameters
import numpy as np
class BasicTDOA():
    def __init__(self, loaded, sample_interval_size, overlap_factor, array_params):
        self.loaded = loaded
        self.sample_interval_size = sample_interval_size
        self.overlap_factor = overlap_factor
        self.prev_weight = 0.0
        self.measurements = self.loaded.get_measurements()
        self.array_params = array_params

    #gcc like gcc_phat only without phat
    def gcc(self, x1, x2):
        x1_ = np.concatenate((x1, np.zeros(len(x2))))
        x2_ = np.concatenate((x2, np.zeros(len(x1))))
        X1 = np.fft.fft(x1_)
        X2 = np.fft.fft(x2_)
        gcc = X2 * np.conj(X1) #Switching so that the first one is reference signal

        gcc = np.fft.ifft(gcc)

        tau = np.argmax(gcc) #- len(sig)
        if tau > len(x1):
            tau -= len(x1)
            tau = -(len(x1) -tau)

        return tau

    #gcc-phat https://reader.elsevier.com/reader/sd/pii/S0921889016304742?token=004CA3D314CB607A43021376366439576DCD31BF6EFE5790BB1779C07EE00D02B7B95F734597DE8A3F13557A198C5D56
    
    
    #could use rfft
    #with padding
    def gcc_phat(self, x1, x2):
        x1_ = np.concatenate((x1, np.zeros(len(x2))))
        x2_ = np.concatenate((x2, np.zeros(len(x1))))
        X1 = np.fft.fft(x1_)
        X2 = np.fft.fft(x2_)
        gcc = X2 * np.conj(X1) #Switching so that the first one is reference signal


        gcc = np.fft.ifft(gcc/abs(gcc))

        tau = np.argmax(gcc) #- len(sig)
        if tau > len(x1):
            tau -= len(x1)
            tau = -(len(x1) -tau)

        return tau

    #Just why is that


    #https://reader.elsevier.com/reader/sd/pii/S0921889016304742?token=004CA3D314CB607A43021376366439576DCD31BF6EFE5790BB1779C07EE00D02B7B95F734597DE8A3F13557A198C5D56
    def calc_max_tau(self, sample_frequency, speed_of_sound, distance):
        return sample_frequency * distance / speed_of_sound


    

    #finds the best tau with the given tau max and the correlation function
    def find_tau(self, correlation_function, x1, x2, start, tau_max, interval_size):

        valid = True
        tau = correlation_function(x1[start : start + interval_size], x2[start : start + interval_size])
        
        if tau > tau_max or tau < - tau_max:
            print("wrong tau detected, error, expected: " + str(tau_max) + " got: " + str(tau))
            valid = False

        return tau, valid


    def calculate_taus(self, correlation_function, interval_overlap, interval_size):
        stamps = []
        taus = []
        valid = []


        #calculate max_taus
        tau_maxes = []
        for i in range(0, len(self.measurements)):
            tau_maxes.append([])
            for j in range(0, len(self.measurements)):
                if i== j:
                    tau_max = 0

                else:
                    dist = self.array_params.get_distance(i, j)
    
                    tau_max = self.calc_max_tau(self.loaded.get_meta_data()["sampling_rate"], 343, dist)
        
                    #Aufrunden, da Tau diskrete Sample steps sind
                    tau_max = math.ceil(tau_max)

                tau_maxes[i].append(tau_max)



        if interval_overlap > 1 or interval_overlap < 0:
            print("interval overlap must be between 0 and 1")
            return None

        if interval_size is None:
            stamps.append(0)
            current_taus = []
            current_valid = []


            
            interval_size = len(self.measurements[0])
            for i in range(0, len(self.measurements)):
                current_taus.append([])
                current_valid.append([])
                for j in range(0, len(self.measurements)):
                    if i == j:
                        tau = 0
                        valid_t = True
                    else:
                        x1 = self.measurements[i]
                        x2 = self.measurements[j]
                        #get tau, using interval of 200 ms
                
                        tau, valid_t = self.find_tau(correlation_function, x1, x2, 0, tau_maxes[i][j], interval_size)
                    
                    current_valid[i].append(valid_t)
                    current_taus[i].append(tau)
            #save taus
            taus.append(current_taus)
            valid.append(current_valid)

        else:
            #compare all microphones to 1; using till -3 tau_max, so that we can fit the last one
            #calculate interval_size in samples
            interval_size_samples = int(math.ceil(self.loaded.get_meta_data()["sampling_rate"] * interval_size))
            for start in np.arange(0, len(self.measurements[0]) + 1 - interval_size_samples, interval_size_samples * (1 - interval_overlap)):
                

                if start + interval_size_samples > len(self.measurements[0]):
                    interval_s = len(self.measurements[0]) - start
                else: 
                    interval_s = interval_size_samples

                #convert start to int by rounding it, int cast because of bug
                r_start = int(round(start))
                #print(r_start)

                stamps.append(r_start)
                current_taus = []
                current_valid = []
                #iterate over other microphones
                for i in range(0, len(self.measurements)):
                    current_taus.append([])
                    for j in range(0, len(self.measurements)):
                        if i == j:
                            tau = 0
                            valid_t = True
                        else:
                            x1 = self.measurements[i]
                            x2 = self.measurements[j]

                            #get tau, using interval of 200 ms
                            tau, valid_t = self.find_tau(correlation_function, x1, x2, r_start, tau_maxes[i][j], interval_s)

                            current_taus[i].append(tau)
                            current_valid[i].append(valid_t)

                #save taus
                taus.append(current_taus)
                valid.append(current_valid)

        return stamps, taus, valid

    def tdoa(self, correlation_function, interval_overlap, interval_size):
        #using the highest possible tau for all microphones, TODO adapt to only use specialised taus for every microphone
        #highest distance
        #distance = 0
        #for i in range(1, len(self.measurements)):
        #    dist = self.array_params.get_distance(0, i)
        #    if dist > distance:
        #        distance = dist
        #
        #tau_max = self.calc_max_tau(self.loaded.get_meta_data()["sampling_rate"], 343, dist)
        
        #Aufrunden, da Tau diskrete Sample steps sind
        #tau_max = math.ceil(tau_max)

        return self.calculate_taus(correlation_function, interval_overlap, interval_size)


    def tdoa_gcc(self, interval_overlap, interval_size=None):
        return self.tdoa(self.gcc, interval_overlap, interval_size)

    def tdoa_gcc_phat(self, interval_overlap, interval_size=None):
        return self.tdoa(self.gcc_phat, interval_overlap, interval_size)