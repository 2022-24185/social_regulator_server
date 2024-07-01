from neat.genome import DefaultGenome
from neat.config import DefaultClassConfig

class GenomeDistanceCache(object):
    def __init__(self, genome_config: DefaultClassConfig):
        self.config = genome_config
        self.distances = {}
        self.hits = 0
        self.misses = 0

    def __call__(self, genome0: DefaultGenome, genome1: DefaultGenome):
        g0 = genome0.key
        g1 = genome1.key
        d = self.distances.get((g0, g1))
        if d is None:
            # Distance is not already computed.
            d = genome0.distance(genome1, self.config)
            self.distances[g0, g1] = d
            self.distances[g1, g0] = d
            self.misses += 1
        else:
            self.hits += 1

        return d