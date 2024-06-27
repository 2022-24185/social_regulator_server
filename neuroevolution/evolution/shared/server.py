# neuroevolution.pureples.shared.server.py

""" The server application for the social brain project."""
# Standard library imports
import pickle
import threading
import time
from typing import Callable, Dict

# Third-party imports
import neat
import neat.config
import neat.genome
from flask import Flask, Response, jsonify, request

# Local application imports
from neuroevolution.evolution.es_hyperneat import ESNetwork
from neuroevolution.evolution.substrate import Substrate
from neuroevolution.evolution.population_evolver import \
    PopulationEvolver

from neuroevolution.evolution.reproduction import \
    MixedGenerationReproduction

from neuroevolution.evolution.species_set import \
    MixedGenerationSpeciesSet

from neuroevolution.evolution.stagnation import \
    MixedGenerationStagnation

# Type aliases
Genome = neat.genome.DefaultGenome
Population = Dict[int, Genome]
Network = neat.nn.RecurrentNetwork
Config = neat.config.Config
FitnessFunction = Callable[[Population, Config], None]

# Constants
INPUT_COORDINATES = [(-0.33, -1.), (0.33, -1.)]
OUTPUT_COORDINATES = [(-0.5, 1.), (0., 1.), (0.5, 1.)]
SUBSTRATE = Substrate(INPUT_COORDINATES, OUTPUT_COORDINATES,)
PARAMS = {"initial_depth": 1,
            "max_depth": 2,
            "variance_threshold": 0.03,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
            "max_weight": 8.0,
            "activation": "sigmoid"}

def calculate_fitness(pop: Population, config: Config) -> None:
    """
    Calculate the fitness of an individual based on user interaction data.
    Data is in the form: 
    {
        'time_since_startup': time.time() - self.start_time,
        'time_since_last_button_click': time.time() - self.last_button_click_time,
        'user_rating': self.rating,
    }
    """
    # Extract the data and calculate the raw fitness scores
    # print in green
    print("\033[92mCalculating fitness...\033[0m")
    time_since_startups = []
    user_ratings = []
    for genome_id, genome in pop.items():
        data = genome.data
        time_since_startups.append(data['time_since_startup'])
        user_ratings.append(data['user_rating'])
    print("\033[92mNormalizing...\033[0m")
    # Normalize the time_since_startup and user_rating
    min_time, max_time = min(time_since_startups), max(time_since_startups)
    min_rating, max_rating = min(user_ratings), max(user_ratings)
    for genome_id, genome in pop.items():
        normalized_time = (genome.data['time_since_startup'] - min_time) / (max_time - min_time)
        normalized_rating = (genome.data['user_rating'] - min_rating) / (max_rating - min_rating)

        # Combine the normalized time_since_startup and user_rating to calculate the fitness
        genome.fitness = normalized_time + normalized_rating
    # print genome id with their fitnesses
    print("\033[92mFitnesses: ", {genome_id: genome.fitness for genome_id, genome in pop.items()}, "\033[0m")
    print("done calculating fitness")

def create_network_from_genome(genome: Genome, config: Config) -> Network:
    """
    Create a network from a genome.
    """
    cppn = neat.nn.FeedForwardNetwork.create(genome, config)
    network = ESNetwork(SUBSTRATE, cppn, PARAMS)
    return network.create_phenotype_network()

def pickle_network(genome_key: int, network: Network) -> bytes:
    """
    Pickle a network.
    """
    return pickle.dumps((genome_key, network))

def create_app() -> Flask:
    """
    Create the Flask application.
    """
    app = Flask(__name__)
    app.debug = True
    stats = neat.statistics.StatisticsReporter()
    config = neat.config.Config(neat.genome.DefaultGenome, MixedGenerationReproduction,
                            MixedGenerationSpeciesSet, MixedGenerationStagnation,
                            'neuroevolution/social_brain/config_cppn_social_brain')
    pop = PopulationEvolver(config, calculate_fitness)
    pop.add_reporter(stats)
    pop.start_reporting()

    @app.route('/test', methods=['GET'])
    def test_route() -> Response:
        genome = pop.pop_manager.get_random_available_genome()
        if genome is not None:
            net = create_network_from_genome(genome, config)
            pickled_net = pickle_network(genome.key, net)
            return Response(pickled_net, mimetype='application/octet-stream')
        return jsonify({'message': 'No individuals available!'}), 200

    @app.route('/user_data', methods=['POST'])
    def receive_user_data() -> Response:
        # Get the JSON data sent by the client
        data = request.get_json()
        # make population handle the data
        pop.handle_receive_user_data(data)
        # print data in pink
        print("\033[95mReceived user data: ", data, "\033[0m")
        return jsonify({'message': 'User data received successfully'}), 200
    return app


def main_loop() -> None:
    """
    Main loop of the application.
    """
    while True:
        print("Waiting for user request...")
        time.sleep(10)


def main() -> None:
    """
    Main function of the application.
    """
    app = create_app()
    threading.Thread(target=main_loop).start()
    app.run()


if __name__ == "__main__":
    main()