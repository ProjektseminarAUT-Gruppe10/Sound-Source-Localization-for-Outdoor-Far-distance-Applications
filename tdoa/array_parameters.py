import math

class ArrayParameters():
    def __init__(self, positions):
        self.positions = positions # array of tuples/ arrays

    def get_distance(self, i1, i2):
        return math.sqrt(math.pow(self.positions[i1][0] - self.positions[i2][0], 2) + math.pow(self.positions[i1][1]- self.positions[i2][1], 2))

    def get_position(self, i):
        return self.positions[i]