# neuroevolution/pureples/shared/evolution.py
import pickle, logging, base64
import neat
from typing import Callable, Dict, TYPE_CHECKING

import neat.genome
from neuroevolution.evolution.tournament_population import TournamentPopulation
from neuroevolution.evolution.tournament_reproduction import TournamentReproduction
from neuroevolution.evolution.tournament_species import TournamentSpeciesSet
from neuroevolution.evolution.tournament_stagnation import TournamentStagnation
from neuroevolution.evolution.substrate import Substrate
from neuroevolution.evolution.es_hyperneat import ESNetwork

if TYPE_CHECKING:
    from neuroevolution.server.models import UserData

# Type aliases
Genome = neat.genome.DefaultGenome
Population = Dict[int, neat.genome.DefaultGenome]
Network = neat.nn.RecurrentNetwork
Config = neat.config.Config
FitnessFunction = Callable[[Population, Config], None]

# Constants
# interpolate so we have 3 THREEE coordinates
INPUT_COORDINATES = [(-0.33, -1.), (0, 0), (0.33, 1.)]
OUTPUT_COORDINATES = [(-0.5, 1.), (0., -1.), (0., -1.), (0.5, 1.)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES,)
PARAMS = {
    "initial_depth": 1,
    "max_depth": 2,
    "variance_threshold": 0.03,
    "band_threshold": 0.3,
    "iteration_level": 1,
    "division_threshold": 0.5,
    "max_weight": 8.0,
    "activation": "sigmoid"
}

class EvolutionManager:
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = neat.Config(
            neat.DefaultGenome,
            TournamentReproduction,
            TournamentSpeciesSet,
            TournamentStagnation,
            self.config_path
        )
        self.population = TournamentPopulation(self.config, self.calculate_fitness)

    def calculate_fitness(self, pop: Population, config: Config) -> None:
        print("\033[92mCalculating fitness...\033[0m")
        time_since_startups = []
        user_ratings = []
        # we want to grab this item: 
        # self.population[user_data.genome_id].data = user_data
        for genome_id, genome in pop.items():
            data = genome.data
            time_since_startups.append(data.time_since_startup)
            user_ratings.append(data.user_rating)
        print("\033[92mNormalizing...\033[0m")
        min_time, max_time = min(time_since_startups), max(time_since_startups)
        min_rating, max_rating = min(user_ratings), max(user_ratings)
        for genome_id, genome in pop.items():
            normalized_time = (genome.data.time_since_startup - min_time + 1) / (max_time - min_time + 1)
            normalized_rating = (genome.data.user_rating - min_rating + 1) / (max_rating - min_rating + 1)
            genome.fitness = normalized_time + normalized_rating
        print("\033[92mFitnesses: ", {genome_id: genome.fitness for genome_id, genome in pop.items()}, "\033[0m")
        print("done calculating fitness")

    def create_network_from_genome(self, genome: Genome, config: Config) -> Network:
        cppn = neat.nn.FeedForwardNetwork.create(genome, config)
        network = ESNetwork(SUBSTRATE, cppn, PARAMS)
        return network.create_phenotype_network()

    def pickle_network(self, genome_key: int, network: Network) -> bytes:
        return pickle.dumps((genome_key, network))

    def fetch_new_genome(self):
        """Fetch a new genome from the population."""
        logging.info("Fetching new genome")
        try:
            genome = self.population.get_random_non_evaluated_member()
            logging.info(f"Fetched genome {genome.key}")
            return genome # id is inside genome.key
        except RuntimeError:
            return None
        
    def get_mediator(self, genome):
        return self.create_network_from_genome(genome, self.config)
    
    def get_serialized_network(self, genome : Genome) -> bytes:
        pickled = self.pickle_network(genome.key, self.get_mediator(genome))
        return base64.b64encode(pickled).decode('utf-8')

    def receive_evaluation(self, user_data: 'UserData'):
        logging.info(f"Received evaluation for genome {user_data.genome_id}")
        self.population.receive_evaluation(user_data)

    def advance_population(self):
        self.population.advance_population()

    def reset(self):
        self.population = TournamentPopulation(self.config, self.calculate_fitness)
