import loader
import basic_tdoa
import array_parameters
import array_default

Loaded = loader.Loader("../measurements/measurement_2/", "A1_CH_00_10")


array_params = array_default.ArrayDefault()
Tdoa = basic_tdoa.BasicTDOA(Loaded, 200, 0.5, array_params)

#print(Tdoa.gcc_phat(Loaded.measurements[0], Loaded.measurements[1]))

print(Tdoa.tdoa_gcc_phat(0.5))

Tdoa.loaded.filter_butter_band_pass(200, 2400)

print(Tdoa.tdoa_gcc_phat(0.5))