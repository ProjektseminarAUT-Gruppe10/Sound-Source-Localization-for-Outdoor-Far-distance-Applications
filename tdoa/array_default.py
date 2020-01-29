
import array_parameters

class ArrayDefault(array_parameters.ArrayParameters):
    def __init__(self):
        positions = [[0, 0.5], [0.3536, 0.3536], [0.5, 0], [0.3536, -0.3536], [0, -0.5], [-0.3536, -0.3536], [-0.5, 0], [-0.3536, 0.3536]]
        super().__init__(positions)