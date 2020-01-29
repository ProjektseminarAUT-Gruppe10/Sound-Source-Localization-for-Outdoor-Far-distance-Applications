import array_parameters

class ArrayUp(array_parameters.ArrayParameters):
    def __init__(self):
        positions = [[0, 0.35], [0.2475, 0.2475], [0.35, 0], [0.2475, -0.2475], [0, 0.35], [-0.2475, -0.2475], [-0.35, 0], [-0.2475, 0.2475]]
        super().__init__(positions)