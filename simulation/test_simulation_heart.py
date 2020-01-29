import simulation
import math
import matplotlib.pyplot as plt

def heart(x):
    return math.pow(math.sin(x), 63) * math.sin(x + 1.5) * 8

source_position = [0, 0]
microphone_positions = [[1, 0], [-1, 0]]
amplitude = 400.0
sample_rate = 48
num_samples = 1600
signal_function = heart

reflection_points = [[100, 1], [-100, 1], [50, 1], [-50, 1], [2, 1], [-2, 1]]

sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position, reflection_points=reflection_points)
sim.calculate()
sim.visualize_signals()

reflection_points = []
sim = simulation.Simulation(amplitude, microphone_positions=microphone_positions, sample_rate=sample_rate, num_samples=num_samples, signal_function=signal_function, source_position=source_position, reflection_points=reflection_points)
sim.calculate()
sim.visualize_signals()




