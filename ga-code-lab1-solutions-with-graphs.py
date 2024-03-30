# This code implements a very simple GA solving the "methinks it is like a weasel" problem.

import random                   # from random, "seed", "choice", "sample", "randrange", "random" are used
import statistics               # from statistics "mean", "stdev" are used
import matplotlib as mpl        # matplotlib is used to generate the nice plots
import matplotlib.pyplot as plt # matplotlib is used to generate the nice plots
from t_test import t_test as t_test

# For the xkcd style the following parameters need to be set
# To get the full effect the Humor Sans font for matplotlib needs to be used:
# Humor Sans is here: https://github.com/shreyankg/xkcd-desktop/blob/master/Humor-Sans.ttf

from matplotlib import patheffects
mpl.rcParams['font.family'] = ['Humor Sans']
mpl.rcParams['font.size'] = 14.0
mpl.rcParams['path.sketch'] = (1, 100, 2)
mpl.rcParams['path.effects'] = [patheffects.withStroke(linewidth=4, foreground="w")]
mpl.rcParams['axes.linewidth'] = 1.5
mpl.rcParams['lines.linewidth'] = 2.0
mpl.rcParams['figure.facecolor'] = 'white'
mpl.rcParams['grid.linewidth'] = 0.0
mpl.rcParams['axes.unicode_minus'] = True
mpl.rcParams['axes.prop_cycle'] = mpl.cycler(color=['b', 'r', 'c', 'm'])
mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['xtick.major.width'] = 3
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['ytick.major.width'] = 3

# Initialises the GA population
#   Fills an empty population array with N=pop_size random individuals
#   Each individual is represented by a Python dictionary with two elements: "solution" and "fitness"
#     each "fitness" is initialised to <None> as the associated solution has not yet been assessed
#     each "solution" is initialised to a string of random symbols from the alphabet
#     either each "solution" is the same random string (converged=True) or
#     each "solution" is a different random string (converged=False)
#     the function as provided doesn't implement the converged=True functionality

def initialise(pop_size, genome_length, genetic_alphabet, converged=False):

    pop = []

    if converged:
        solution = "".join(random.choice(genetic_alphabet) for _ in range(genome_length))
        while len(pop)<pop_size:
            pop.append({"fitness":None, "solution":solution})
    else:
        while len(pop)<pop_size:
            solution = "".join(random.choice(genetic_alphabet) for _ in range(genome_length))
            pop.append({"fitness":None, "solution":solution})

    return pop


# Counts the number of locations for which two strings of the same length match.
#   E.g, matches of "red", "rod" should be 2.

def matches(str1, str2):
    return sum([str1[i]==str2[i] for i in range(len(str1))])


# Assesses the fitness of each individual in the current population
#   For each individual, the number of symbols in the solution that match the target string are counted 
#   Stores this as the fitness of the individual (normalised by the target string length)
#   Maximum fitness is thus 1 (all symbols match); minimum fitness is 0 (no matches).
#   Sorts the population by fitness with the best solution at the top of the list

def assess(pop, target):

    length = len(target)

    for i in pop:
        i["fitness"] = matches(i["solution"], target) / length

    return sorted(pop, key = lambda i: i["fitness"], reverse=True)    # <<< *important


# Tournament selection is run to pick a parent solution
#   A sample of tournament_size unique indivduals from the current population is considered 
#   The solution belonging to the winner (the individual with the highest fitness) is returned

def tournament(pop, tournament_size):

    competitors = random.sample(pop, tournament_size)
    winner = competitors.pop()

    while competitors:
        i=competitors.pop()
        if i["fitness"] > winner["fitness"]:
            winner = i

    return winner["solution"]


# Breeds a new generation of solutions from the existing population
#   Generates N offspring solutions from a population of N individuals
#   Chooses parents with a bias towards those with higher fitness
#   Tournament selection is applied
#   'Elitism' is employed which means the current best individual
#       always gets copied into the next generation at least once
#   'Crossover' (uniform or single point) can be used which combines
#       2 parent genotypes into 1 offspring

def breed(pop, tournament_size, crossover, uniform, elitism):

    offspring_pop = []

    if elitism:
        elite = pop[0]
        offspring_pop.append({"fitness":None, "solution":elite["solution"]})

    while len(offspring_pop)<len(pop):
        mum = tournament(pop, tournament_size)
        if random.random()<crossover:                                           
            dad = tournament(pop, tournament_size)                              
            offspring_pop.append({"fitness":None, "solution":cross(mum, dad)})  
        else:
            offspring_pop.append({"fitness":None, "solution":mum})            

    return offspring_pop


# Mutation to the population of new offspring is applied
#   Each symbol in each solution may be replaced by a randomly chosen symbol from the alphabet
#   For each symbol in each solution the chance of this happening is set by the mutation parameter

def mutate(pop, mutation, alphabet, elitism):

    length = len(pop[0]["solution"])

    for i in pop[elitism:]:               
        for j in range(length):
            if random.random()<mutation:
                i["solution"] = i["solution"][:j] + random.choice(alphabet) + i["solution"][j+1:]

    return pop


# Crossover the solution string of two parents to make an offspring
#   (This code implements 'one-point crossover')
#   Picks a random point in the solution string,
#       uses the mum's string up to this point and the dad's string after it

def cross(mum, dad):
    point = random.randrange(len(mum))
    return mum[:point] + dad[point:]


# Uniform crossover of two parent solution strings to make an offspring
#   picks each offspring solution symbol from the mum or dad with equal probability

def uniform_cross(mum, dad):
    return "".join( mum[i] if random.choice([True, False]) else dad[i] for i in range(len(mum)) )


# Multi-point crossover of two parent solution strings to make an offspring
#   Picks n random points in the solution string,
#       uses the mum's string up to the first point and the dad's string up to the second point, etc.

def multi_point_cross(mum, dad, n):

    genomes = [mum,dad]
    crosses = random.sample(range(len(mum)),n)
    parent = i = 0
    offspring = ""

    while i<len(mum):
        offspring+=genomes[parent][i]
        if i in crosses:
            parent = 1-parent
        i+=1

    return offspring


# Writes a line of summary stats for population pop at generation gen
#   if File is None write to the standard out, otherwise write to the File

def write_fitness(pop, gen, file=None):

    fitness = [p["fitness"] for p in pop]

    # calculates how different the best current solution is from the current worst and the current median
    #   because the population is sorted by fitness, the current 'best' is item [0] in the population,
    #       the current worst is the last item [-1], and the 'median' individual is half-way down the list

    max_diff = len(pop[0]["solution"]) - matches(pop[0]["solution"],pop[-1]["solution"])
    med_diff = len(pop[0]["solution"]) - matches(pop[0]["solution"],pop[int(len(pop)/2)]["solution"])
    line = "{:4d}: max:{:.3f}, min:{:.3f}, mean:{:.3f}, stdev:{:.3f}, max_diff:{:2d}, med_diff:{:2d}".format(gen,max(fitness),min(fitness),statistics.mean(fitness),statistics.stdev(fitness),max_diff,med_diff)

    if file:
        file.write(line+"\n")
    else:
        print(line)


# The main function for the GA
#  The function takes a number of arguments specifying various parameters and options
#  each argument has a default value which can be overloaded in the function call..
#   Seeds the pseudo-random number generator (using the system clock)
#     so no two runs will have the same sequence of pseudo-random numbers
#   Sets the length of the solution strings to be the length of the target string
#   Sets the mutation rate to be equivalent to "on average 1 mutation per offspring"
#   Initialises a population of individuals
#   Assesses each member of the initial population using the fitness function
#   Runs a maximum of max_gen generations of evolution
#     (stopping early if we find the perfect solution)
#   Each generation of evolution comprises:
#     increments the generation counter
#     breeds a new population of offspring
#     mutates the new offspring
#     assesses each member of the new population using the fitness function and sort pop by fitness
#     tracks the best (highest fitness) solution in the current population (the 0th item in the list)
#   Returns the final generation count and the best individual from the final population
        
def do_the_ga(pop_size=100, tournament_size=2, crossover=0.0, uniform=False,
              elitism=False, max_gen=1000, converged=False, write_every=1, file=None,
              target="methinks it is like a weasel", m=1.0, 
              alphabet="abcdefghijklmnopqrstuvwxyz "):

    random.seed()

    length = len(target)
    mutation =   m/length   
    pop = initialise(pop_size, length, alphabet, converged)
    pop = assess(pop, target)
    generation = 0
    best = pop[0]

    while generation < max_gen and best["fitness"] < 1:

        generation += 1
        pop = breed(pop, tournament_size, crossover, uniform,elitism)
        pop = mutate(pop, mutation, alphabet, elitism)
        pop = assess(pop, target)
        best = pop[0]

        if write_every and generation % write_every==0:
            write_fitness(pop, generation, file)

    return generation, best

def simple_main():

    # calls the Genetic Algorithm (GA) with the default set up, receives the final generation count and the final best individual
    gens, best = do_the_ga()
    print("{:4d} generations yielded: '{}' ({:.3f})".format(gens,best["solution"],best["fitness"]))
    input("hit return to continue")

    # calls the Genetic Algorithm (GA) with the default set up (but writes stats to a file), receivse the final generation count and the final best individual
    with open("ga_output.dat",'w') as f:
        gens, best = do_the_ga(file=f)
        print("{:4d} generations yielded: '{}' ({:.3f})".format(gens,best["solution"],best["fitness"]))
    input("hit return to continue")

    # calls the Genetic Algorithm (GA) with a longer target string than the default set up
    gens, best = do_the_ga(target="something quite a bit longer than methinks it is like a weasel", write_every=0)
    print("{:4d} generations yielded: '{}' ({:.3f})".format(gens,best["solution"],best["fitness"]))
    input("hit return to continue")


# Example of how to explore the impact of varying a parameter on the performance of our GA
def batch_main():

    # opens a file to store the results in...
    with open("ga_output_max_gen_100_to_6400.dat",'w') as f:

        # loops over different parameter values for population size to see what difference it makes..
        for max_gen in [100, 200, 400, 800, 1600, 3200, 6400]:

            # calls the Genetic Algorithm (GA) - tells it to only write out stats for the final generation of evolution..
            gens, best = do_the_ga(max_gen=max_gen, file=f, write_every=max_gen)
            print("With max_gen={:4d}, {:4d} generations yielded: '{}' ({:.3f})".format(max_gen, gens, best["solution"], best["fitness"]))


# simple_main()
# batch_main()

# Q1. This code executes the Standard Genetic Algorithm (GA) once and assesses the quality of the final evolved solution. 
# It determines if the solution is perfect and the number of generations required to achieve it, alongside evaluating the performance range.
# The assessment includes considering appropriate performance measures such as average, best, and worst scores.
# It also explores measures of variability, diversity, and predictability, including the range (max score - min score), interquartile range, and standard deviation.
# The code evaluates the fairness of aggregating data from runs that reached the solution with those that did not.
            
def q1():

    gens, best = do_the_ga()

    print("{:4d} generations yielded: '{}' ({:.3f})".format(gens,best["solution"],best["fitness"]))
    input("[Hit return to continue]\n")

    # assesses the range of performance we can call the do_the_ga() function several times
    # saves time by setting write_every to 0 to prevent the printing of progress during each run

    performance = []
    runs=50

    for r in range(runs):
        gens, best = do_the_ga(write_every=0)
        print("run {:2d}: {:4d} generations yielded: '{}' ({:.3f})".format(r, gens, best["solution"], best["fitness"]))
        performance.append(best["fitness"])

    plt.title("Range of performance of the Standard GA")
    plt.xlabel("Best Score In 1000th Generation")
    plt.ylabel("Number of Runs")
    plt.hist(performance)
    plt.show()

    input("[Hit return to continue]\n")



# ts (tournament size) makes a big difference - big ts improves search
#   when ts is very large evolution is not entirely implemented - it is similar to hill climbing
#   when ts is 1 it is performing random search
    
def q2a():

    quality = []
    speed = []
    ts = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 40, 80, 100]

    for size in ts:
        gens, best = do_the_ga(tournament_size=size,write_every=0)
        print("With tournament_size={:4d}, {:2d} generations yielded: '{}' ({:.3f})".format(size, gens, best["solution"], best["fitness"]))
        quality.append(best["fitness"])
        speed.append(gens)

    plt.title("Impact of Tournament Size\non Solution Quality")
    plt.xlabel("Tournament Size")
    plt.ylabel("Final Fitness")
    plt.xscale('log')
    plt.plot(ts, quality)
    plt.show()

    input("[Hit return to continue]\n")

    plt.title("Impact of Tournament Size\non Time To Find Perfect Solution")
    plt.xlabel("Tournament Size")
    plt.ylabel("Final Generation #")
    plt.xscale('log')
    plt.plot(ts, speed)
    plt.show()

    input("[Hit return to continue]\n")


# target length doesn't seem to make much difference if the rest of the parameters are standard
#  a "floor effect" is observed - evolution finds all of the targets hard
#  if tournament size is set to 3 then target length starts to make a difference
    
def q2b():

    target = "methinks it is like a weasel but longer than "
    quality = []
    speed = []
    lengths = [x for x in range(5,101,5)] + [x for x in range(200,1001,100)]

    for length in lengths:

        t = "".join(target[i%len(target)] for i in range(length))
        gens, best = do_the_ga(target=t,tournament_size=3,write_every=0)

        print("With target={}, {:2d} generations yielded: '{}' ({:.3f})".format(t, gens, best["solution"], best["fitness"]))

        quality.append(best["fitness"])
        speed.append(gens)

    plt.title("Impact of Target Length\non Solution Quality")
    plt.xlabel("Target Length")
    plt.ylabel("Final Fitness")
    plt.xscale('log')
    plt.plot(lengths, quality)
    plt.show()

    input("[Hit return to continue]\n")

    plt.title("Impact of Target Length\non Time To Find Perfect Solution")
    plt.xlabel("Target Length")
    plt.ylabel("Final Generation #")
    plt.xscale('log')
    plt.plot(lengths, speed)
    plt.show()

    input("[Hit return to continue]\n")

    cm = plt.cm.get_cmap('RdYlBu')

    plt.title("Impact of Target Length\non Evolutionary Performance")
    plt.xlabel("Final Fitness")
    plt.ylabel("Final Generation #")

    sc = plt.scatter(quality, speed, c=lengths, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel('Target Length', rotation=90)

    plt.show()

    input("[Hit return to continue]\n")


# alphabet size seems to also not make much difference if the selection pressure is too weak (ts=2)
#  again this is a "floor effect" - if ts=3 or ts=4 a clearer picture is seen
#  if tournament size is set to 4 then alphabet size starts to slow evolution down
#  does speed decrease linearly with increasing alphabet?
    
def q2c():

    for alphabet in ["abcdefghijklmnopqrstuvwxyz ", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ ", "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!£$%^&*()'[]{}@:~#;<>,./?|\`¬ "]:
        gens, best = do_the_ga(alphabet=alphabet,tournament_size=4,write_every=0)
        print("With alphabet={:2d}, {:2d} generations yielded: '{}' ({:.3f})".format(len(alphabet), gens, best["solution"], best["fitness"]))

    input("[Hit return to continue]\n")

    quality = []
    speed = []
    symbols = "abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!£$%^&*()'[]{}@:~#;<>,./?|\`¬"
    lengths = range(27,97)

    for a in lengths:

        alphabet = symbols[:a]
        gens, best = do_the_ga(alphabet=alphabet, tournament_size=4, write_every=0)

        print("With alphabet={:2d}, {:2d} generations yielded: '{}' ({:.3f})".format(len(alphabet), gens, best["solution"], best["fitness"]))

        quality.append(best["fitness"])
        speed.append(gens)

    plt.title("Impact of Alphabet Length\non Time To Find Perfect Solution")
    plt.xlabel("Target Length")
    plt.ylabel("Final Generation #")
    plt.plot(lengths, speed)
    plt.show()

    input("[Hit return to continue]\n")

# pop size has a straightforward positive effect on performance - but shows up differently for tournament size=2 or 3 or 4
    
def q2d():

    size_list = [10, 20, 40, 80, 160, 320, 640, 1280]
    quality = []
    speed = []

    for pop_size in size_list:

        gens, best = do_the_ga(pop_size=pop_size,tournament_size=3,write_every=0)

        print("With pop_size={:3d}, {:2d} generations yielded: '{}' ({:.3f})".format(pop_size, gens, best["solution"], best["fitness"]))

        quality.append(best["fitness"])
        speed.append(gens)

    plt.title("Impact of Population Size\non Time To Find Perfect Solution")
    plt.xlabel("Population Size")
    plt.ylabel("Final Generation #")
    plt.plot(size_list, speed)
    plt.show()

    input("[Hit return to continue]\n")


# a change to the do_the_ga program is needed
#  an easy one is to change the m=1/length command to mutation = m/length where m is a parameter that is passed to the function
#  very low mutation rates can reach the solution but take longer
#  high mutation rates fail to settle on the solution
#  varying mutation rate changes mutation selection balance by reducing/increasing mutation pressure
#  mutation = 0 allows no search
#  mutation = 1 allows no inheritance, randomly sampling the space - a
    
def q2e():

    m_list = [m/100.0 for m in range(0, 200, 5)]+ [m for m in range(2, 30)]
    quality = []
    speed = []

    for m in m_list:

        gens, best = do_the_ga(m=m, write_every=0)

        print("With m={:.2f}, {:2d} generations yielded: '{}' ({:.3f})".format(m, gens, best["solution"], best["fitness"]))

        quality.append(best["fitness"])
        speed.append(gens)

    plt.title("Impact of Mutation Rate\non Solution Quality")
    plt.xlabel("Mutations per Offspring Genotype")
    plt.ylabel("Final Fitness")
    plt.xscale('log')
    plt.plot(m_list, quality)
    plt.show()

    input("[Hit return to continue]\n")

    plt.title("Impact of Mutation Rate\non Time To Find Perfect Solution")
    plt.xlabel("Mutations per Offspring Genotype")
    plt.ylabel("Final Generation #")
    plt.xscale('log')
    plt.plot(m_list, speed)
    plt.show()

    input("[Hit return to continue]\n")

    cm = plt.cm.get_cmap('RdYlBu')

    plt.title("Impact of Mutation Rate\non Evolutionary Performance")
    plt.xlabel("Final Fitness")
    plt.ylabel("Final Generation #")

    sc = plt.scatter(quality, speed, c=m_list, cmap=cm)
    cbar = plt.colorbar(sc)
    cbar.ax.set_ylabel("Mutations per Offspring Genotype", rotation=90)

    plt.show()

    input("[Hit return to continue]\n")


# Elitism improves the performance of the GA
    
def q3a():

    elitism_quality = []
    no_elitism_quality = []

    for run in range(50):

        elitism=True
        gens, best = do_the_ga(elitism=elitism,write_every=0)

        print("Run {:2d}: With Elitism={}, {:2d} generations yielded: '{}' ({:.3f})".format(run, elitism, gens, best["solution"], best["fitness"]))
        
        elitism_quality.append(best["fitness"])

    for run in range(50):

        elitism=False
        gens, best = do_the_ga(elitism=elitism,write_every=0)
        
        print("Run {:2d}: With Elitism={}, {:2d} generations yielded: '{}' ({:.3f})".format(run, elitism, gens, best["solution"], best["fitness"]))
        
        no_elitism_quality.append(best["fitness"])

    plt.title("Impact of Elitism\non Solution Quality")
    plt.xlabel("Best Score In 1000th Generation")
    plt.ylabel("Number of Runs")
    plt.hist([elitism_quality, no_elitism_quality], alpha=0.5, label=['Elitism', 'No_Elitism'])
    plt.legend(loc='upper left')
    plt.show()

    input("[Hit return to continue]\n")

# Converged or random initial population doesn't make any difference...
# This is because any initial diversity in the population collapses quickly
#   wiping out the difference between the two scenarios.
# This can be shown by plotting a measure of convergence over time.
    
def q3b():

    initially_converged = []
    initially_random = []

    for run in range(50):

        converged=True
        gens, best = do_the_ga(converged=converged,tournament_size=3,write_every=0)

        print("Run {:2d}: With Converged={}, {:2d} generations yielded: '{}' ({:.3f})".format(run, converged, gens, best["solution"], best["fitness"]))

        initially_converged.append(gens)
        converged=False
        gens, best = do_the_ga(converged=converged,tournament_size=3,write_every=0)

        print("Run {:2d}: With Converged={}, {:2d} generations yielded: '{}' ({:.3f})".format(run, converged, gens, best["solution"], best["fitness"]))

        initially_random.append(gens)

    plt.title("Impact of Initial Convergence\non Time To Find Perfect Solution")
    plt.xlabel("Number of Generations")
    plt.ylabel("Number of Runs")
    plt.hist([initially_converged,initially_random], rwidth=0.5, bins=10, alpha=0.5, label=['Initially Converged','Initially Random'])
    plt.legend(loc='upper right')
    plt.show()

    input("[Hit return to continue]\n")

# Crossover straightforwardly improves performance
# Uniform crossover might be a bit better? It might need several runs and a t-test?
    
def q3c():

    crossover_list = [i/20.0 for i in range(21)]
    uniform_crossover_quality=[]
    one_pt_crossover_quality=[]
    uniform_crossover_speed=[]
    one_pt_crossover_speed=[]

    for crossover in crossover_list:

        gens, best = do_the_ga(crossover=crossover,write_every=0)

        print("With 1-point crossover={:.2f}, {:2d} generations yielded: '{}' ({:.3f})".format(crossover, gens, best["solution"], best["fitness"]))

        one_pt_crossover_speed.append(gens)
        one_pt_crossover_quality.append(best["fitness"])
        gens, best = do_the_ga(crossover=crossover,uniform=True,write_every=0)

        print("With uniform crossover={:.2f}, {:2d} generations yielded: '{}' ({:.3f})".format(crossover, gens, best["solution"], best["fitness"]))

        uniform_crossover_speed.append(gens)
        uniform_crossover_quality.append(best["fitness"])

    plt.title("Impact of Crossover\non Solution Quality")
    plt.ylabel("Final Fitness")
    plt.xlabel("Probability of Crossover")
    plt.plot(crossover_list,one_pt_crossover_quality, label='One-Point Crossover')
    plt.plot(crossover_list,uniform_crossover_quality, label='Uniform Crossover')
    plt.legend(loc='lower right')
    plt.show()

    input("[Hit return to continue]\n")

    plt.title("Impact of Crossover\non Time To Find Perfect Solution")
    plt.ylabel("Final Generation #")
    plt.xlabel("Probability of Crossover")
    plt.plot(crossover_list,one_pt_crossover_speed, label='One-Point Crossover')
    plt.plot(crossover_list,uniform_crossover_speed, label='Uniform Crossover')
    plt.legend(loc='upper right')
    plt.show()

    input("[Hit return to continue]\n")

#  It is shown how to look at how convergence changes over time for four different values of mutation probability
#  In each case compared are how fitness changes over evolutionary time with how solution diversity changes over time
#  "Diversity" could be measured in many ways but an easy one is how different is the current best solution from the current median solution (using the matches() function)
#  The resulting graphs should show that:
#    very low mutation rate (0.05/L) means rapid loss of diversity in the population (="prematurely convergence") resulting in very slow progress
#    very high mutation rate (5/L) means very high diversity in the population because high mutation prevents heritability of good genes and hamstrings selection
#    intermedite mutation rate (0.05/L or 1/L) allows moderate diversity that is lost as the population gets closer to the optimal solution
    
def q4():

    # opens a file to store the results in...

    with open("ga_output_mut_convergence_data.dat",'w') as f:

        # loops over different parameter values for mutation rate to see what difference it makes..

        for m in [0.05, 0.5, 1.0, 5.0]:
            gens, best = do_the_ga(m=m, write_every=1, file=f, max_gen=500) # calls the Genetic Algorithm (GA) with the right parameters
            print("With m={:.2f}, {:2d} generations yielded: '{}' ({:.3f})".format(m, gens, best["solution"], best["fitness"]))

input("Run q1... [hit return to start]")
q1()

input("Run q2a... [hit return to start]")
q2a()

input("Run q2b... [hit return to start]")
q2b()

input("Run q2c... [hit return to start]")
q2c()

input("Run q2d... [hit return to start]")
q2d()

input("Run q2e... [hit return to start]")
q2e()

input("Run q3a... [hit return to start]")
q3a()

input("Run q3b... [hit return to start]")
q3b()

input("Run q3c... [hit return to start]")
q3c()

input("Run q4... [hit return to start]")
q4()

print("done")