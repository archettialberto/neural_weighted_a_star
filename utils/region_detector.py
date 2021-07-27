import numpy as np


class RegionDetector:
    def __init__(self, x_max, y_max, sources_per_target, f):
        self.sources_per_target = sources_per_target
        self.x_max, self.y_max = x_max, y_max
        self.f = f
        fx, fy = int(x_max * f), int(y_max * f)

        self.regions = {
            'nw': [0, 0, fx, y_max // 2],
            'ne': [0, y_max // 2, fx, y_max],
            'sw': [x_max - fx, 0, x_max, y_max // 2],
            'se': [x_max - fx, y_max // 2, x_max, y_max],
            'en': [0, y_max - fy, x_max // 2, y_max],
            'es': [x_max // 2, y_max - fy, x_max, y_max],
            'wn': [0, 0, x_max // 2, fy],
            'ws': [x_max // 2, 0, x_max, fy],
        }

        self.opposite_regions = {
            'nw': ['se', 'sw', 'es'],
            'ne': ['se', 'sw', 'ws'],
            'sw': ['ne', 'nw', 'en'],
            'se': ['ne', 'nw', 'wn'],
            'en': ['wn', 'ws', 'sw'],
            'es': ['wn', 'ws', 'nw'],
            'wn': ['en', 'es', 'se'],
            'ws': ['en', 'es', 'ne'],
        }

    def sample_journey(self):
        region, opposite = self.sample_region()
        target = self.sample_from_region(region)
        sources = []
        while len(sources) < self.sources_per_target:
            source = self.sample_from_region(opposite)
            if source not in sources:
                sources.append(source)
        return target, sources

    def __sample_point(self):
        x = np.random.randint(0, self.x_max)
        y = np.random.randint(0, self.y_max)
        return x, y

    def sample_region(self):
        keys = [k for k in self.regions.keys()]
        index = np.random.randint(0, len(keys))
        return [keys[index]], self.opposite_regions[keys[index]]

    def sample_from_region(self, region):
        x, y = self.__sample_point()
        while not self.is_in_region((x, y), region):
            x, y = self.__sample_point()
        return x, y

    def is_in_region(self, point, region) -> bool:
        px, py = point
        for r in region:
            rx_min, ry_min, rx_max, ry_max = self.regions[r]
            if rx_min <= px < rx_max and ry_min <= py < ry_max:
                return True
        return False

    def print_points(self, points):
        a = np.zeros((self.x_max, self.y_max))
        for p in points:
            px, py = p
            a[px, py] = 1
        print(a)
