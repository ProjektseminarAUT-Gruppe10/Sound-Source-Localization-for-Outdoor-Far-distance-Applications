import loader
import matplotlib.pyplot as plt
import numpy as np

l = loader.Loader("../measurements/measurement_2/", "A1_CH_00_10")

meta_data = l.get_meta_data()

signal = l.measurements[0]

complexFourierTransform = np.fft.fft(signal)
frequencies_x = np.fft.fftfreq(signal.size, d=meta_data["sampling_spacing"])

amplitude_y   = np.abs(complexFourierTransform)
phase_y       = np.angle(complexFourierTransform)

plt.subplot(2,1,1)
plt.ylabel("Amplitudengang")
plt.xlabel("Frequency in Hz")
plt.plot(frequencies_x,amplitude_y)



l.filter_butter_band_pass(900, 1000)
signal = l.measurements[0]

complexFourierTransform = np.fft.fft(signal)
frequencies_x = np.fft.fftfreq(signal.size, d=meta_data["sampling_spacing"])

amplitude_y   = np.abs(complexFourierTransform)
phase_y       = np.angle(complexFourierTransform)

plt.subplot(2,1,2)
plt.ylabel("Amplitudengang")
plt.xlabel("Frequency in Hz")
plt.plot(frequencies_x,amplitude_y)


plt.show()

