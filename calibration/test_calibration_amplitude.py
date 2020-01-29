import calibrator
import matplotlib.pyplot as plt
import os, sys
tdoa_path = os.path.join('..', 'tdoa')
sys.path.append(tdoa_path)

power_path = os.path.join('..', 'libraries')
sys.path.append(power_path)

import SignalProcessingLibrary
import GeometryLibrary
import loader


def run_test(distance, angle, start, end, plot_):
    if angle < 10:
        angle_str = "0" + str(int(angle))
    else:
        angle_str = str(int(angle))
    if distance< 10:
        distance_str = "0" + str(int(distance))
    else:
        distance_str = str(int(distance))

    stri = "A2_CH_" + angle_str + '_' + distance_str
    loaded = loader.Loader("../measurements/measurement_1/", stri)

    signals = loaded.get_measurements()


    # cut them to the beginning of a signal
    if end < 0:
        end = len(signals[0])

    for i in range(0, len(signals)):
        signals[i] = signals[i][start:end]


    #also plot them
    if plot_:
        plt.figure()

        x = range(0, len(signals[0]))
        for i in range(0, len(signals)):
            plt.subplot(8, 1, i + 1)

            plt.plot(x, signals[i])

        plt.show()

    #main calculation
    array = GeometryLibrary.calculateMicrophoneArray_2(0.35, GeometryLibrary.getPoint(0, 0))
    source_pos = GeometryLibrary.getPoint(0, distance)

    distances = [] #TODO calculate them
    for i in range(0, len(signals)):
        distances.append(GeometryLibrary.distance(source_pos, array[i]))
        #print(distances[i])


    speed_of_sound = 343.3
    sample_rate = loaded.get_meta_data()["sampling_rate"]
    calib = calibrator.Calibrator(signals, distances, speed_of_sound, sample_rate)

    ks = calib.run_calibration_amplitude()

    print(ks)
    #s = calib.signals
    #for i in range(0, 8):
    #    print(len(s[i]))


def main():
    start = 0
    end = -1
    
    distance = 10
    angle = 0
    start = 170000
    end = 270000
    plot_ = True
    run_test(distance, angle, start, end, plot_)

    distance = 20
    angle = 0
    start = 80000 
    end = 180000
    plot_ = True
    run_test(distance, angle, start, end, plot_)

    distance = 40
    angle = 0
    start = 80000
    end = 180000
    plot_ = True
    run_test(distance, angle, start, end, plot_)

    distance = 60
    angle = 0
    start = 65000
    end = 165000
    plot_ = True
    run_test(distance, angle, start, end, plot_)


main()