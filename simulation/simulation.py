import math
import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, amplitude=1.0, 
            microphone_positions=[], 
            sample_rate=48000, 
            num_samples=1000, 
            signal_function=None, 
            source_position=None, 
            microphone_noise_sigmas=None, 
            microphone_noise_mus=None,
            microphone_noise_amplitudes=None, 
            general_noise_sigma=None, 
            general_noise_mu=None,
            general_noise_amplitude=None, 
            source_noise_sigma=None, 
            source_noise_mu=None, 
            reflection_points=[], 
            seed=None,
            damping_factor=1.0):
        self.speed_of_sound = 343.3

        if not seed is None:
            np.random.seed(seed)

        self.amplitude = amplitude
        self.damping_factor = 1.0
        if signal_function is None:
            self.signal_function = lambda x: math.sin(x * (2 * math.pi * 440.0)) if x > 0 else 0
        else:
            self.signal_function = signal_function
        
        if source_position is None:
            self.source_position = [0, 0]
        else: 
            self.source_position = source_position

        self.microphone_positions = microphone_positions

        self.sample_rate = sample_rate
        self.num_samples = num_samples
        if microphone_noise_sigmas is None:
            self.microphone_noise_sigmas = []

            for __ in self.microphone_positions:
                self.microphone_noise_sigmas.append(None)
        else:
            self.microphone_noise_sigmas = microphone_noise_sigmas

        if microphone_noise_mus is None:
            self.microphone_noise_mus = []

            for __ in self.microphone_positions:
                self.microphone_noise_mus.append(None)
        else:
            self.microphone_noise_mus = microphone_noise_mus

        if microphone_noise_amplitudes is None:
            self.microphone_noise_amplitudes = []

            for __ in self.microphone_positions:
                self.microphone_noise_amplitudes.append(None)
        else:
            self.microphone_noise_mus = microphone_noise_mus

        self.general_noise_sigma = general_noise_sigma
        self.general_noise_mu = general_noise_mu
        self.general_noise_amplitude = general_noise_amplitude

        self.source_noise_sigma = source_noise_sigma
        self.source_noise_mu = source_noise_mu

        self.reflection_points = reflection_points

        self.signals = []
        self.microphone_noises = []

        for __ in self.microphone_positions:
            self.signals.append([])
            self.microphone_noises.append([])

        self.general_noise = []
        self.source_noise = []

    def add_microphone(self, position, noise_mu=None, noise_sigma=None):
        self.microphone_positions.append(position)

        self.microphone_noise_mus.append(noise_mu)
        self.microphone_noise_sigmas.append(noise_sigma)


    def precalc_noises(self):
        if not (self.source_noise_mu is None or self.source_noise_sigma is None):
            self.source_noise = np.random.normal(self.source_noise_mu, self.source_noise_sigma, self.num_samples)

        if not (self.general_noise_mu is None or self.general_noise_sigma is None):
            self.general_noise = np.random.normal(self.general_noise_mu, self.general_noise_sigma, self.num_samples)

        for i in range(0, len(self.microphone_positions)):
            if not (self.microphone_noise_mus[i] is None or self.microphone_noise_sigmas[i] is None):
                self.microphone_noises[i] = np.random.normal(self.microphone_noise_mus[i], self.microphone_noise_sigmas[i], self.num_samples)

    # Note this must be continuous, so we assume that before t= 0, no source and no noise was active. This must be treated in the source function as well
    def source_noise_f(self, x):
        if x < 0:
            return 0.0
        pos = int(round(x))
        return self.source_noise[pos]

    def calculate(self):
        self.precalc_noises()

        for sample in range(0, self.num_samples):
            for mic_i in range(0, len(self.microphone_positions)):
                self.signals[mic_i].append(self.calculate_signal_at_microphone(sample, mic_i))

        return self.signals

    def calculate_signal_at_microphone(self, sample, mic_i):
        distance = self.get_distance_to_source(mic_i)
        
        signal = self.calculate_signal_at_distance(1.0, sample, mic_i, distance)

        for reflection_point in self.reflection_points:
            distance = self.get_distance_to_source_reflected(mic_i, reflection_point)

            reflection = self.calculate_signal_at_distance(self.damping_factor, sample, mic_i, distance)
            signal += reflection

        return signal

    def calculate_signal_at_distance(self, factor, sample, mic_i, distance):
        delta_t = distance / self.speed_of_sound
        #Use distance discrete?
        signal = factor * self.signal_function((sample) / self.sample_rate - delta_t)
        

        if not len(self.source_noise) == 0:
            signal += self.source_noise_f(sample - delta_t * self.sample_rate)

        signal = signal * 1 / distance

        if not len(self.general_noise) == 0:
            signal += self.general_noise[sample] * self.general_noise_amplitude

        if not len(self.microphone_noises[mic_i]) == 0:
            signal += self.microphone_noises[mic_i][sample] * 1 # TODO!! MULT WITH AMP

        return signal


    def get_distance_to_source(self, mic_i):
        return math.sqrt(math.pow(self.source_position[0] - self.microphone_positions[mic_i][0], 2) + math.pow(self.source_position[1] - self.microphone_positions[mic_i][1], 2))

    def get_distance_to_source_reflected(self, mic_i, reflection_point):
        d1 = math.sqrt(math.pow(self.source_position[0] - reflection_point[0], 2) + math.pow(self.source_position[1] - reflection_point[1], 2))
        d2 = math.sqrt(math.pow(reflection_point[0] - self.microphone_positions[mic_i][0], 2) + math.pow(reflection_point[1] - self.microphone_positions[mic_i][1], 2))
        return d1 + d2

    def visualize_signals(self):
        fig = plt.figure()
        fig.suptitle("Simulation")
        for mic_i in range(0, len(self.microphone_positions)):
            self.draw_signal(mic_i, fig.add_subplot(len(self.microphone_positions), 1, mic_i + 1))

        plt.show()

    def draw_signal(self, mic_i, sub_p):
        x = range(0, self.num_samples)
        y = self.signals[mic_i]

        sub_p.plot(x, y)
        #sub_p.stem(x, y, use_line_collection=True)
        sub_p.set_title("Microphone: " + str(mic_i + 1))




