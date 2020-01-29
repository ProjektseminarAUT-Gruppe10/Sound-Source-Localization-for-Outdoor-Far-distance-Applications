# Imports
import json
import sys
import math
import numpy as np

sys.path.append("..\\simulation")
import simulation_loader, simulation

sys.path.append("..\\libraries")
from GeometryLibrary import getPoint, estimateK_Pair, distance, getAngle_Pair, getAngle_angle1, angle_degree, getIntersectionPointsCircle, KarstenDOA_calculateCurve_linear
from GraphicLibrary import drawPoint, drawCircle
from SignalProcessingLibrary import getSignalPower_UsingTime_AverageFree

sys.path.append("..\\tdoa")
import array_parameters
import basic_tdoa

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
    sim = simulation.Simulation(config["amplitude"],
                                microphone_positions=config["microphone_positions"],
                                sample_rate=config["sample_rate"],
                                num_samples=config["number_samples"],
                                signal_function=signal_function,
                                source_position=config["source_position"],
                                microphone_noise_mus=config["microphone_noise_mus"],
                                microphone_noise_sigmas=config["microphone_noise_sigmas"],
                                microphone_noise_amplitudes=config["microphone_noise_amplitudes"],
                                general_noise_sigma=config["general_noise_sigma"],
                                general_noise_mu=config["general_noise_mu"],
                                source_noise_mu=config["source_noise_mu"],
                                source_noise_sigma=config["source_noise_sigma"],
                                reflection_points=config["reflection_points"],
                                general_noise_amplitude=config["general_noise_amplitude"],
                                damping_factor=config["damping_factor"])

    signals = sim.calculate()    
    loaded = simulation_loader.SimulationLoader(signals, config["sample_rate"])
    
    return loaded

def generateTime(sampling_rate, number_samples):
    return np.arange(0,number_samples)/sampling_rate

def convertPoint(p):
    return getPoint(p[0],p[1])

def signal_function(x): 
    return math.sin(x * (2 * math.pi * 400.0)) if (x > 0.1 and x < 0.3) else 0
        
def getPlausibleOne(a,b):
    if(b[1]<0):
        return a
    elif(a[1]<0):
        return b
    else:
        if(np.linalg.norm(a)<np.linalg.norm(b)):
            return b
        else:
            return a
        
# Lade Konfiguration
config = load_configs("config.json")[0]

for i in range(0,10):
    config["source_position"][0] = -10
    config["source_position"][1] = (i+1)*10
    
    # Starte Simulation
    loaded = simulate(config)
    signals = loaded.get_measurements()
    meta    = loaded.get_meta_data()
    
    # Source Signal
    time = generateTime(meta["sampling_rate"],meta["number_samples"])
    source_signal = np.zeros_like(time)
    for t in range(0,time.shape[0]):
        source_signal[t] = signal_function(time[t])
        
    # Do Power Calculations
    powers = list()
    for s in signals:
        z = np.asarray(s)
        power = getSignalPower_UsingTime_AverageFree(np.asarray(s))
        powers.append(power)
    #    print("microphone ",power,"W")
    #print("Source Power ",getSignalPower_UsingTime_AverageFree(source_signal),"W")
    
    # Calculate TDOA
    arr = array_parameters.ArrayParameters(config["microphone_positions"])
    tdoa = basic_tdoa.BasicTDOA(loaded, 0, 0.0, arr)
    
    # Do Amplitude Verfahren
    powerA = powers[0]
    powerB = powers[1]
    micA_pos = convertPoint(config["microphone_positions"][0])
    micB_pos = convertPoint(config["microphone_positions"][1])
    dTDOA = tdoa.tdoa_gcc_phat(0.0)[1][0][1][0]
    dSDOA = dTDOA*meta["sampling_spacing"]*343.2
    
    K1, K2 = estimateK_Pair(powerA, powerB, micA_pos, micB_pos, dSDOA)
    rA_1 = K1/np.sqrt(powerA)
    rA_2 = K2/np.sqrt(powerA)
    rB_1 = K1/np.sqrt(powerB)
    rB_2 = K2/np.sqrt(powerB)
    
    solutions = getIntersectionPointsCircle(micA_pos, rA_2, micB_pos, rB_2)
    estimPos = getPlausibleOne(solutions[0],solutions[1])
    sourcePos = convertPoint(config["source_position"])

    print(i, config["source_position"],estimPos," ",sourcePos," > ",distance(estimPos,sourcePos))
