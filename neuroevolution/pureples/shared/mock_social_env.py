"""
i want to make this file #editor into a testbed for my social interaction bot, having the same role as the gym_runner file, but with the possibility to interact with the visual interface #file:visual.py . some suggested components: 
- **Population Initializer:** Initializes a population of neural network individuals.
- **Fitness Evaluator:** Connects to the interactive interface, allowing users to evaluate the fitness of individuals.
- **Neural Network Activator:** Triggers the individual's neural network activation every 10 seconds.
- **Action Selector:** Decides the action based on the neural network output.
"""

import time
import neat
from neat import Population, DefaultGenome, DefaultReproduction, DefaultSpeciesSet, DefaultStagnation
from flask import Flask, jsonify
import queue
import threading


# Initialize population
def ini_pop(state, stats, config, output):
    """
    Initialize population attaching statistics reporter.
    """
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def evaluate_fitness(individual):
    # Connect to interactive interface and allow users to interact with the bot
    # Return a fitness score based on the interaction
    #fitness_score = connect_to_interactive_interface_and_interact_with_bot()
    #return fitness_score
    pass

def activate_network(individual, state):
    # Pass the current state of the bot through the individual's neural network
    # Return the output of the network
    pass

def select_action(network_output):
    # Decide an action based on the network output
    pass

# Create a thread-safe queue
individual_queue = queue.Queue()

def handle_user_request():
    # This function runs in a separate thread for each user
    # Get an individual from the queue
    try:
        individual = individual_queue.get_nowait()
    except queue.Empty:
        print("No more individuals available")
        return

    # Use the individual to interact with the user
    # ...




# while True:
#     # Evaluate fitness of each individual in the population
#     for individual in pop.population.values():
#         individual.fitness = evaluate_fitness(individual)

#     # Activate neural network of each individual and select an action
#     for individual in pop.population.values():
#         state = None  # Get the current state of the bot
#         network_output = activate_network(individual, state)
#         action = select_action(network_output)
#         # Perform the action

#     # Run every 10 seconds
#     time.sleep(10)

def main():
    # Initialize your variables here
    state = None
    stats = None
    config = None
    output = None

    app = Flask(__name__)
    #pop = ini_pop(None, None, None, None)

    @app.route('/test', methods=['GET'])
    # Call your functions here
    #pop = ini_pop(state, stats, config, output)
    #evaluate_fitness(pop)
    def test_route():
        return jsonify({'message': 'Flask is working!'}), 200

    def main_loop():
        while True:
            print("Waiting for user request...")

            # Run every 10 seconds
            time.sleep(10)
    
    # Start the main loop in a separate thread
    threading.Thread(target=main_loop).start()

    # Start the Flask app
    app.run()


if __name__ == "__main__":
    main()