import numpy as np
import matplotlib.pyplot as plt
import randomized_optimization as ro


from mlrose.opt_probs import TSPOpt, DiscreteOpt
from mlrose.fitness import TravellingSales, Knapsack, FourPeaks


def Knapsack_f(length, weights, values, random_seeds):
    Knapsack_objective = Knapsack(weights, values, max_weight_pct=0.35)
    problem = DiscreteOpt(length=length, fitness_fn=Knapsack_objective, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=1000, sa_max_iters=1000, ga_max_iters=250, mimic_max_iters=50,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.05, 2.01, 0.05), sa_min_temp=0.001,
                          ga_pop_size=300, mimic_pop_size=1500, ga_keep_pct=0.2, mimic_keep_pct=0.4,
                          pop_sizes=np.arange(100, 2001, 200), keep_pcts=np.arange(0.1, 0.81, 0.1),
                          plot_name='Knapsack', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot performances for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=1000, sa_max_iters=1000, ga_max_iters=500, mimic_max_iters=100,
                         sa_init_temp=100, sa_exp_decay_rate=0.020, sa_min_temp=0.001,
                         ga_pop_size=100, ga_keep_pct=0.2,
                         mimic_pop_size=300, mimic_keep_pct=0.4,
                         plot_name='Knapsack', plot_ylabel='Fitness')


def four_peaks(length, random_seeds):

    # Define Four Peaks objective function and problem
    four_fitness = FourPeaks(t_pct=0.1)
    problem = DiscreteOpt(length=length, fitness_fn=four_fitness, maximize=True, max_val=2)
    problem.set_mimic_fast_mode(True)  # set fast MIMIC

    # Plot optimizations for RHC, SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=100, sa_max_iters=1000, ga_max_iters=50, mimic_max_iters=50,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.002, 0.04, 0.002), sa_min_temp=0.001,
                          ga_pop_size=1000, mimic_pop_size=1000, ga_keep_pct=0.1, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(100, 1001, 100), keep_pcts=np.arange(0.1, 0.51, 0.1),
                          plot_name='Four Peaks', plot_ylabel='Fitness')

    # Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot performances for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=1000, sa_max_iters=5000, ga_max_iters=100, mimic_max_iters=100,
                         sa_init_temp=100, sa_exp_decay_rate=0.02, sa_min_temp=0.001,
                         ga_pop_size=1000, ga_keep_pct=0.1,
                         mimic_pop_size=1000, mimic_keep_pct=0.2,
                         plot_name='Four Peaks', plot_ylabel='Fitness')




def travel_salesman(length, distances, random_seeds):

    # Define Travel Salesman objective function and problem
    fitness_dists = TravellingSales(distances=distances)
    # Define optimization problem object
    problem = TSPOpt(length=length, fitness_fn=fitness_dists , maximize=False)
    problem.set_mimic_fast_mode(True)  # set fast MIMIC

    # Plot optimizations for SA, GA and MIMIC
    print('\nPlot Optimizations for SA, GA and MIMIC')
    ro.plot_optimizations(problem=problem,
                          random_seeds=random_seeds,
                          rhc_max_iters=500, sa_max_iters=500, ga_max_iters=50, mimic_max_iters=50,
                          sa_init_temp=100, sa_decay_rates=np.arange(0.005, 0.05, 0.005), sa_min_temp=0.001,
                          ga_pop_size=100, mimic_pop_size=100, ga_keep_pct=0.2, mimic_keep_pct=0.2,
                          pop_sizes=np.arange(100, 401, 100), keep_pcts=np.arange(0.1, 0.51, 0.1),
                          plot_name='TSP', plot_ylabel='Fitness')

    #Plot performances for RHC, SA, GA and MIMIC
    print('\nPlot Performances for RHC, SA, GA and MIMIC')
    ro.plot_performances(problem=problem,
                         random_seeds=random_seeds,
                         rhc_max_iters=500, sa_max_iters=500, ga_max_iters=500, mimic_max_iters=500,
                         sa_init_temp=100, sa_exp_decay_rate=0.02, sa_min_temp=0.001,
                         ga_pop_size=100, ga_keep_pct=0.15,
                         mimic_pop_size=300, mimic_keep_pct=0.5,
                         plot_name='TSP', plot_ylabel='Fitness')


if __name__ == "__main__":

    random_seeds = [5 + 5 * i for i in range(2)]  # random seeds for get performances over multiple random runs


    # Create list of distances between pairs of cities
    cities_distances = [(0, 1, 3.1623), (0, 2, 4.1231), (0, 3, 5.8310), (0, 4, 4.2426), (0, 5, 5.3852), \
                 (0, 6, 4.0000), (0, 7, 2.2361), (1, 2, 1.0000), (1, 3, 2.8284), (1, 4, 2.0000), \
                 (1, 5, 4.1231), (1, 6, 4.2426), (1, 7, 2.2361), (2, 3, 2.2361), (2, 4, 2.2361), \
                 (2, 5, 4.4721), (2, 6, 5.0000), (2, 7, 3.1623), (3, 4, 2.0000), (3, 5, 3.6056), \
                 (3, 6, 5.0990), (3, 7, 4.1231), (4, 5, 2.2361), (4, 6, 3.1623), (4, 7, 2.2361), \
                 (5, 6, 2.2361), (5, 7, 3.1623), (6, 7, 2.2361)]

    # Experiment the Travel Salesman Problem, Flip Flop and Four Peaks with RHC, SA, GA and MIMIC
    travel_salesman(length=8, distances=cities_distances, random_seeds=random_seeds)
    #mlrose example
    weights = [10, 5, 2, 8, 15]
    values = [1, 2, 3, 4, 5]
    Knapsack_f(length=5, weights=weights, values=values, random_seeds=random_seeds)
    four_peaks(length=100, random_seeds=random_seeds)
