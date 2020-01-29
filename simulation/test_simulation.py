import simulation
import math


source_position = [-5, 0]
microphone_positions = [[0, 0], [5, 0]]
amplitude = 40.0
sample_rate = 48000
num_samples = 2200
signal_function = lambda x: math.sin(x * (2 * math.pi * 440.0)) if x > 0 else 0

sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position)

print(sim.calculate())

sim.visualize_signals()


