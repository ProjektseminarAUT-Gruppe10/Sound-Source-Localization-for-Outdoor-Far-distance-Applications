import simulation
import math


source_position = [-10, 0]
microphone_positions = [[0, 0], [10, 0]]
amplitude = 40.0
sample_rate = 48000
num_samples = 220

microphone_noises_sigma = [1.0, 1.0]
microphone_noises_mu = [0.0, 0.0]
microphone_noises_amplitudes = [1.0, 1.0]

source_noise_sigma = 1.0
source_noise_mu = 0.0

general_noise_sigma = 1.0
general_noise_mu = 0.0
general_noise_amplitude = 1.0

reflection_points = [[5, 5]]


signal_function = lambda x: math.sin(x * (2 * math.pi * 440.0))

sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position, microphone_noise_mus=microphone_noises_mu, microphone_noise_sigmas=microphone_noises_sigma, general_noise_sigma=general_noise_sigma, general_noise_mu=general_noise_mu, source_noise_mu=source_noise_mu, source_noise_sigma=source_noise_sigma, reflection_points=reflection_points, microphone_noise_amplitudes=microphone_noises_amplitudes, general_noise_amplitude=general_noise_amplitude)

print(sim.calculate())

sim.visualize_signals()


