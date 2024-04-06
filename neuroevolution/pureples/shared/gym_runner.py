"""
Generic runner for AI Gym - runs Neat, Hyperneat and ES-Hyperneat
"""

import neat
import numpy as np
from neuroevolution.pureples.hyperneat.hyperneat import create_phenotype_network
from neuroevolution.pureples.es_hyperneat.es_hyperneat import ESNetwork


def ini_pop(state, stats, config, output):
    """
    Initialize population attaching statistics reporter.
    """
    pop = neat.population.Population(config, state)
    if output:
        pop.add_reporter(neat.reporting.StdOutReporter(True))
    pop.add_reporter(stats)
    return pop


def run_es(gens, env, max_steps, config, params, substrate, max_trials=100, output=True):
    """
    Generic OpenAI Gym runner for ES-HyperNEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):

        for _, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(substrate, cppn, params)
            net = network.create_phenotype_network()

            fitnesses = []

            for _ in range(trials):
                ob, _ = env.reset()
                net.reset()

                total_reward = 0

                for _ in range(max_steps):
                    for _ in range(network.activations):
                        o = net.activate(ob)

                    action = np.argmax(o)
                    ob, reward, terminated, _, _ = env.step(action)
                    total_reward += reward
                    if terminated:
                        break

                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run_hyper(gens, env, max_steps, config, substrate, activations, max_trials=100,
              activation="sigmoid", output=True):
    """
    Generic OpenAI Gym runner for HyperNEAT.
    """
    trials = 1
 
    def eval_fitness(genomes, config):
        print("EVALUATING FITNESS")

        for count, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            net = create_phenotype_network(cppn, substrate, activation)

            fitnesses = []

            for i in range(trials):
                ob, _ = env.reset()
                if(count < 2):
                    print("Trial", i, "ob:", ob)
                net.reset()

                total_reward = 0

                for othercount in range(max_steps):
                    for thirdcount in range(activations):
                        o = net.activate(ob)
                        if(count < 2):
                            if(othercount < 2):
                                if(thirdcount < 2):
                                    print("Output", o)
                    action = np.argmax(o)
                    ob, reward, done, _, _ = env.step(action) # step returns Tuple[ObsType, float, bool, bool, dict]
                    if(count < 2):
                        if(othercount < 2):
                                print("Reward", reward)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)


def run_neat(gens, env, max_steps, config, max_trials=100, output=True):
    """
    Generic OpenAI Gym runner for NEAT.
    """
    trials = 1

    def eval_fitness(genomes, config):

        for _, g in genomes:
            net = neat.nn.FeedForwardNetwork.create(g, config)

            fitnesses = []

            for _ in range(trials):
                ob = env.reset()

                total_reward = 0

                for _ in range(max_steps):
                    o = net.activate(ob)
                    action = np.argmax(o)
                    ob, reward, done, _ = env.step(action)
                    total_reward += reward
                    if done:
                        break
                fitnesses.append(total_reward)

            g.fitness = np.array(fitnesses).mean()

    # Create population and train the network. Return winner of network running 100 episodes.
    stats_one = neat.statistics.StatisticsReporter()
    pop = ini_pop(None, stats_one, config, output)
    pop.run(eval_fitness, gens)

    stats_ten = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0), stats_ten, config, output)
    trials = 10
    winner_ten = pop.run(eval_fitness, gens)

    if max_trials == 0:
        return winner_ten, (stats_one, stats_ten)

    stats_hundred = neat.statistics.StatisticsReporter()
    pop = ini_pop((pop.population, pop.species, 0),
                  stats_hundred, config, output)
    trials = max_trials
    winner_hundred = pop.run(eval_fitness, gens)
    return winner_hundred, (stats_one, stats_ten, stats_hundred)
