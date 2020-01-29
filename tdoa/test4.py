import visualizer
import array_default

array_params = array_default.ArrayDefault()
stamps = [0]
taus = [[[0, 17, 60, 104, 126, 110, 63, 19], [-17, 0, 43, 87, 109, 127, 45, 1], [-60, -43, 0, 44, 83, 122, 2, -152], [-104, -87, -44, 0, -12, 113, 74, 126], [-126, -109, -83, 12, 0, 33, 22, 115], [-110, -127, -122, -113, -33, 0, -44, -87], [-63, -45, -2, -74, -22, 44, 0, -44], [-19, -1, 152, -126, -115, 87, 44, 0]]]
valids = [[[True, True, True, True, True, True, True, True], [True, True, True, True, True, True, True, True], [True, True, True, True, True, False, True, False], [True, True, True, True, True, False, True, True], [True, True, True, True, True, True, True, True], [True, True, False, False, True, True, True, True], [True, True, True, True, True, True, True, True], [True, True, False, True, True, True, True, True]]]
#source_pos = (0, 10)
source_pos = None
v = visualizer.Visualizer(array_params, taus, valids, stamps, source_pos)

v.evaluate()