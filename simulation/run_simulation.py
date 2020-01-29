import simulation
import math
import sys
import json
import os

import simulation_loader
tdoa_path = os.path.join('..', 'tdoa')
sys.path.append(tdoa_path)

import basic_tdoa
import array_parameters

def load_configs(filename):
    default = {
        "position": [
            0,
            0
        ],
        "noise": {
            "sigma": 0,
            "mu": 0,
            "amplitude": 0
        }
    }
    with open(filename) as configs_file:
        datas = json.load(configs_file)

        def get_key(arr, key, default):
            return list(map(lambda e: e.get(key, default), arr))

        configs = []

        for data in datas:
            RUNS = data.get("runs", 0)
            AMPLITUDE = data.get("amplitude", 0)
            DAMPING_FACTOR = data.get("damping_factor", 0)
            sample = data.get("sample", {"rate": 0, "count": 0})
            SAMPLE_RATE = sample.get("rate", 0)
            SAMPLE_COUNT = sample.get("count", 0)
            microphones = data.get("microphones", [default])
            MICROPHONE_POSITIONS = get_key(
                microphones, "position", [default["position"]])
            microphone_noises = get_key(
                microphones, "noise", [default["noise"]])
            MICROPHONE_NOISES_SIGMA = get_key(microphone_noises, "sigma", [
                default["noise"]["sigma"]])
            MICROPHONE_NOISES_MU = get_key(microphone_noises, "mu", [
                default["noise"]["mu"]])
            MICROPHONE_NOISES_AMPLITUDE = get_key(microphone_noises, "amplitude", [
                default["noise"]["amplitude"]])
            reflections = data.get("reflections", [default])
            REFLECTION_POSITIONS = get_key(
                reflections, "position", [default["position"]])
            reflection_noises = get_key(
                reflections, "noise", [default["noise"]])
            REFLECTION_NOISES_SIGMA = get_key(reflection_noises, "sigma", [
                default["noise"]["sigma"]])
            REFLECTION_NOISES_MU = get_key(reflection_noises, "mu", [
                default["noise"]["amplitude"]])
            REFLECTION_NOISES_AMPLITUDE = get_key(reflection_noises, "amplitude", [
                default["noise"]["amplitude"]])
            sound_sources = data.get("sound_sources", [default])
            SOURCE_POSITIONS = get_key(
                sound_sources, "position", [default["position"]])
            source_noises = get_key(sound_sources, "noise", [default["noise"]])
            SOURCE_NOISES_SIGMA = get_key(source_noises, "sigma", [
                default["noise"]["sigma"]])
            SOURCE_NOISES_MU = get_key(source_noises, "mu", [
                default["noise"]["mu"]])
            SOURCE_NOISES_AMPLITUDE = get_key(source_noises, "amplitude", [
                default["noise"]["amplitude"]])
            diffuse = data.get("diffuse", default)
            diffuse_noise = diffuse.get("noise", default["noise"])
            DIFFUSE_NOISE_SIGMA = diffuse_noise.get(
                "sigma", default["noise"]["sigma"])
            DIFFUSE_NOISE_MU = diffuse_noise.get(
                "sigma", default["noise"]["mu"])
            DIFFUSE_NOISE_AMPLITUDE = diffuse_noise.get(
                "sigma", default["noise"]["amplitude"])
            SEED = data.get("seed", 0)

            configs.append({
                "runs": RUNS,
                "amplitude": AMPLITUDE,
                "damping_factor": DAMPING_FACTOR,
                "sample_rate": SAMPLE_RATE,
                "number_samples": SAMPLE_COUNT,
                "microphone_positions": MICROPHONE_POSITIONS,
                "microphone_noise_sigmas": MICROPHONE_NOISES_SIGMA,
                "microphone_noise_mus": MICROPHONE_NOISES_MU,
                "microphone_noise_amplitudes": MICROPHONE_NOISES_AMPLITUDE,
                "reflection_points": REFLECTION_POSITIONS,
                "source_position": SOURCE_POSITIONS[0],
                "source_noise_sigma": SOURCE_NOISES_SIGMA[0],
                "source_noise_mu": SOURCE_NOISES_MU[0],
                "source_noise_amplitude": SOURCE_NOISES_AMPLITUDE[0],
                "general_noise_sigma": DIFFUSE_NOISE_SIGMA,
                "general_noise_mu": DIFFUSE_NOISE_MU,
                "general_noise_amplitude": DIFFUSE_NOISE_AMPLITUDE,
                "seed": SEED
            })

    return configs


def simulate(config):
    def signal_function(x): return math.sin(
        x * (2 * math.pi * 440.0)) if (x > 0.1 and x < 0.3) else 0

    sim = simulation.Simulation(config["amplitude"],
                                microphone_positions=config["microphone_positions"],
                                sample_rate=config["sample_rate"],
                                num_samples=config["number_samples"],
                                signal_function=signal_function,
                                source_position=config["source_position"],
                                microphone_noise_mus=config["microphone_noise_mus"],
                                microphone_noise_sigmas=config["microphone_noise_sigmas"],
                                general_noise_sigma=config["general_noise_sigma"],
                                general_noise_mu=config["general_noise_mu"],
                                source_noise_mu=config["source_noise_mu"],
                                source_noise_sigma=config["source_noise_sigma"],
                                reflection_points=config["reflection_points"],
                                microphone_noise_amplitudes=config["microphone_noise_amplitudes"],
                                general_noise_amplitude=config["general_noise_amplitude"],
                                damping_factor=config["damping_factor"])

    signals = sim.calculate()
    sim.visualize_signals()
    
    loaded = simulation_loader.SimulationLoader(signals, config["sample_rate"])
    arr = array_parameters.ArrayParameters(config["microphone_positions"])

    tdoa = basic_tdoa.BasicTDOA(loaded, 0, 0.0, arr)

    return tdoa.tdoa_gcc_phat(0.0)



def simulateKevin(config):
    def signal_function(x): return math.sin(
        x * (2 * math.pi * 440.0)) if (x > 0.1 and x < 0.3) else 0

    sim = simulation.Simulation(config["amplitude"],
                                microphone_positions=config["microphone_positions"],
                                sample_rate=config["sample_rate"],
                                num_samples=config["number_samples"],
                                signal_function=signal_function,
                                source_position=config["source_position"],
                                microphone_noise_mus=config["microphone_noise_mus"],
                                microphone_noise_sigmas=config["microphone_noise_sigmas"],
                                general_noise_sigma=config["general_noise_sigma"],
                                general_noise_mu=config["general_noise_mu"],
                                source_noise_mu=config["source_noise_mu"],
                                source_noise_sigma=config["source_noise_sigma"],
                                reflection_points=config["reflection_points"],
                                microphone_noise_amplitudes=config["microphone_noise_amplitudes"],
                                general_noise_amplitude=config["general_noise_amplitude"],
                                damping_factor=config["damping_factor"])

    signals = sim.calculate()
    sim.visualize_signals()
    
    loaded = simulation_loader.SimulationLoader(signals, config["sample_rate"])
    arr = array_parameters.ArrayParameters(config["microphone_positions"])

    return loaded, arr
