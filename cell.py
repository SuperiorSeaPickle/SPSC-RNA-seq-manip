from shapely.geometry import Polygon



class cell:
    def __init__(self, id, coords):
        self.id = id
        self.boundry = Polygon(coords)