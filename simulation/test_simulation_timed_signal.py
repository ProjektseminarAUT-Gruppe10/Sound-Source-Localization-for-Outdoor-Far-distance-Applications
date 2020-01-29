import simulation
import math


def timed_sin(x):
    if x < 0:
        return 0.0
    else: 
        return math.sin(x * (2 * math.pi * 440.0))

source_position = [-10, 0]
microphone_positions = [[0, 0], [10, 0]]
amplitude = 40.0
sample_rate = 48000
num_samples = 5500
signal_function = timed_sin

sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position)

print(sim.calculate())

sim.visualize_signals()


