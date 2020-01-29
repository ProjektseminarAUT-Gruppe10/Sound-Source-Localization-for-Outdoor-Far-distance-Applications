import simulation
import math


source_position = [0, 0]
microphone_positions = [[1, 0], [0, 1]]
amplitude = 40.0
sample_rate = 48000
num_samples = 220
signal_function = lambda x: math.sin(x * (2 * math.pi * 440.0))
reflection_points = [[1, 1]]

sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position, reflection_points=reflection_points)

print(sim.calculate())

sim.visualize_signals()


