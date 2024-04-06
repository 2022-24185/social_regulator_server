"""
i want to make this file #editor into a testbed for my social interaction bot, having the same role as the gym_runner file, but with the possibility to interact with the visual interface #file:visual.py . some suggested components: 
- **Population Initializer:** Initializes a population of neural network individuals.
- **Fitness Evaluator:** Connects to the interactive interface, allowing users to evaluate the fitness of individuals.
- **Neural Network Activator:** Triggers the individual's neural network activation every 10 seconds.
- **Action Selector:** Decides the action based on the neural network output.
"""
# Standard library imports
import pickle
import time
import queue
import threading

# Third-party imports
import neat
from neat import Population, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from flask import Flask, jsonify, Response
import numpy as np

# Local application imports
from neuroevolution.pureples.es_hyperneat.es_hyperneat import ESNetwork
from neuroevolution.pureples.shared.substrate import Substrate

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


def init_population(state, stats, config, output):
    """
    Initialize population attaching statistics reporter.
    """
    population = neat.population.Population(config, state)
    
    # Add a reporter
    if output:
        population.add_reporter(neat.reporting.StdOutReporter(True))
    population.add_reporter(stats)

    return population


def evaluate_fitness(individual):
    # Connect to interactive interface and allow users to interact with the bot
    # Return a fitness score based on the interaction
    #fitness_score = connect_to_interactive_interface_and_interact_with_bot()
    #return fitness_score
    pass

def select_action(network_output):
    # Decide an action based on the network output
    pass

def handle_user_request():
    # This function runs in a separate thread for each user
    # Get an individual from the queue
    try:
        individual = individual_queue.get_nowait()
    except queue.Empty:
        print("No more individuals available")
        return

def main():
    stats = neat.statistics.StatisticsReporter()
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                            neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                            'neuroevolution/social_brain/config_cppn_social_brain')
    output = True

    app = Flask(__name__)
    pop = init_population(None, stats, config, output)
    individual_queue = queue.Queue()

    # Add individual genomes to the queue
    for _, genome in pop.population.items():
        individual_queue.put(genome)

    @app.route('/test', methods=['GET'])
    def test_route():
        if not individual_queue.empty():
            genome = individual_queue.get()
            cppn = neat.nn.FeedForwardNetwork.create(genome, config)
            network = ESNetwork(SUBSTRATE, cppn, PARAMS)
            net = network.create_phenotype_network()
            pickled_net = pickle.dumps(net)
            return Response(pickled_net, mimetype='application/octet-stream')
        else: 
            return jsonify({'message': 'No individuals available!'}), 200

    def main_loop():
        while True:
            print("Waiting for user request...")
            time.sleep(10)
    
    threading.Thread(target=main_loop).start()

    # Start the Flask app
    app.run()


if __name__ == "__main__":
    main()