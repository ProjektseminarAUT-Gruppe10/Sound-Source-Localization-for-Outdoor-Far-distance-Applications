import simulation
import math
import sys

import simulation_loader

sys.path.append('../tdoa')
import basic_tdoa
import array_parameters

source_position = [0, 0]
microphone_positions = [[1, 0], [3, 0], [5, 0]]
amplitude = 40.0
sample_rate = 48000
num_samples = int(0.4 * sample_rate)

signal_function = lambda x: math.sin(x * (2 * math.pi * 440.0)) if (x > 0.1 and x < 0.3)  else 0

microphone_noises_sigma = [1.0, 1.0, 1.0]
microphone_noises_mu = [0.0, 0.0, 0.0]
microphone_noise_amplitudes = [0.1, 0.1, 0.1]

source_noise_sigma = 1.0
source_noise_mu = 0.0

general_noise_sigma = 1.0
general_noise_mu = 0.0
general_noise_amplitude = 0.1

reflection_points = [[5, 5]]

sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position, microphone_noise_mus=microphone_noises_mu, microphone_noise_sigmas=microphone_noises_sigma, general_noise_sigma=general_noise_sigma, general_noise_mu=general_noise_mu, source_noise_mu=source_noise_mu, source_noise_sigma=source_noise_sigma, reflection_points=reflection_points, microphone_noise_amplitudes=microphone_noise_amplitudes, general_noise_amplitude=general_noise_amplitude)

signals = sim.calculate()
sim.visualize_signals()

loaded = simulation_loader.SimulationLoader(signals, sample_rate)
arr = array_parameters.ArrayParameters(microphone_positions) 


tdoa = basic_tdoa.BasicTDOA(loaded, 600, 0.0, arr)

print(tdoa.tdoa_gcc_phat(0.0))


