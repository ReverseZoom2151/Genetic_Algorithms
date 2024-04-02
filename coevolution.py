import random       # uses "seed", "choice", "sample", "randrange", "random", "shuffle"
import statistics   # from statistics "mean", "stdev" are used
import matplotlib as mpl          # matplotlib is used to generate the nice plots
import matplotlib.pyplot as plt   # matplotlib is used to generate the nice plots
import textwrap                   # this is used to format the chart title

# Initializes a GA population by filling an empty population array with N=pop_size individuals.
# Each individual is represented by a dictionary containing "fitness" and "solution":
#   "fitness" is initialized to 0, as the associated solution has not been evaluated yet.
#   "solution" initialization depends on the zeropop flag:
#       If zeropop is True (following Cartlidge & Bullock, 2004), each "solution" is a string of "0"s from the binary alphabet.
#       If zeropop is False (for regular evolution), each "solution" gene is a random symbol from the binary alphabet.

def initialise(pop_size, length, alphabet, zeropop=True):

    pop = []

    while len(pop)<pop_size:
        if zeropop:
            pop.append({"fitness":0, "solution":"0"*length})
        else:
            pop.append({"fitness":0, "solution":"".join([random.choice(list(alphabet)) for _ in range(length)])})

    return pop

# Applies a virulence function to parasite fitness based on the approach in Cartlidge and Bullock (2004).
# This function can reward or punish parasites based on their performance against hosts:
#   1. Normalize all parasite scores by dividing by the current maximum score.
#   2. Apply the virulence function to each parasite's normalized score (x), with a parameter "para_lambda":
#       "para_lambda" ranges from 1.0 (maximum virulence) to 0.5 (minimum virulence).
#       The virulence function is defined as: fitness = (2 * x / para_lambda) - (x ^ 2) / (para_lambda ^ 2).
# Note: If all original fitness values in the parasite population are zero, return unchanged to prevent division by zero during normalization.

def apply_virulence(para_pop, para_lambda):

    max_para_fitness = 0

    for i in range(len(para_pop)):
        if para_pop[i]["fitness"] > max_para_fitness:
            max_para_fitness = para_pop[i]["fitness"]

    if max_para_fitness==0:
        return para_pop

    for i in range(len(para_pop)):

        normalised_score = para_pop[i]["fitness"]/max_para_fitness
        para_pop[i]["fitness"] = (2 * normalised_score / para_lambda) - ( (normalised_score**2)/(para_lambda**2) )

    return para_pop

# Assesses the fitness of each individual host in the current population through competitions with randomly chosen parasites:
#   1. Each host competes against n=num_competitions randomly selected parasites.
#   2. In each competition, the one with more '1's in their solution wins; ties favor the host, and the winner gains a fitness point.
#   3. After all competitions, normalize each individual's fitness by dividing by the number of competitions, yielding scores between 0 and 1.
#   4. If para_lambda is set between 0.5 and 1.0, apply the virulence function to adjust the fitness of parasites based on their performance.
#   5. Finally, sort both populations by their fitness scores, placing the highest scoring individuals at the top.

def assess(host_pop, para_pop, num_competitions, para_lambda):

    for competition in range(num_competitions):
        order = list(range(len(para_pop)))
        random.shuffle(order)

        for i in range(len(host_pop)):

            host = host_pop[i]
            para = para_pop[order[i]]

            if host["solution"].count("1") >= para["solution"].count("1"):
                host["fitness"]+=1
            else:
                para["fitness"]+=1

    for i in range(len(host_pop)):
        host_pop[i]["fitness"]/=num_competitions
        para_pop[i]["fitness"]/=num_competitions

    if para_lambda and 0.5 <= para_lambda <= 1.0 :
        para_pop = apply_virulence(para_pop, para_lambda)

    return sorted(host_pop, key = lambda i: i["fitness"], reverse=True), sorted(para_pop, key = lambda i: i["fitness"], reverse=True)

# Runs tournament selection to pick a parent from the population:
#   1. Samples tournament_size unique individuals from the population.
#   2. Returns the solution of the individual with the highest fitness in the sample, identifying the tournament winner.

def tournament(pop, tournament_size):

    competitors = random.sample(pop, tournament_size)
    winner = competitors.pop()

    while competitors:
        i = competitors.pop()
        if i["fitness"] > winner["fitness"]:
            winner = i

    return winner["solution"]

# Breeds a new generation of solutions from the existing population:
#   1. Generates N offspring solutions from a population of N individuals, maintaining the population size.
#   2. Selects each parent (referred to as "mum") for breeding with a bias towards higher fitness individuals, utilizing tournament selection.
#   3. Avoids elitism and employs asexual reproduction to simplify the process.

def breed(pop, tournament_size):

    offspring_pop = []

    while len(offspring_pop)<len(pop):
        mum = tournament(pop, tournament_size)
        offspring_pop.append({"fitness":0, "solution":mum})

    return offspring_pop

# Applies mutation to the population of new offspring, potentially using a mutation bias:
#   1. Iterates over each symbol in each solution, mutating it based on the mutation_rate parameter.
#   2. If mutation occurs, the choice of new symbol from the alphabet may be influenced by a bias parameter:
#       The bias is a list corresponding to the alphabet, dictating the likelihood of choosing each symbol.
#       For example, an alphabet of "01" with bias=["0.5", "0.5"] results in equal chances for "0" and "1".
#       A bias=["0.2", "0.8"] for the same alphabet means "1" is chosen four times more often than "0".
#       With bias=["0.0", "1.0"], "1" is always selected.
#   3. By default, each symbol in the alphabet has an equal chance of being chosen if mutated.

def mutate(pop, length, mutation_rate, alphabet, bias=None):

    if bias==None:
        bias = [1.0/len(alphabet)]*len(alphabet)

    for i in pop:
        for j in range(length):
            if random.random()<mutation_rate:
                i["solution"] = i["solution"][:j] + random.choices(alphabet,bias)[0] + i["solution"][j+1:]

    return pop

# Writes out a line of summary statistics for the host and parasite populations:
#   1. If no file is specified, outputs to standard out; otherwise, writes to the specified file.
#   2. Adds the current objective fitness and relative fitness of all hosts and parasites to global lists,
#       preparing data for future graph plotting.
# This process includes capturing and organizing key performance metrics for both populations to facilitate
#   comprehensive analysis and visualization of evolutionary dynamics over time.

def write_fitness(hosts, paras, gen, file=None):

    global host_gen, host_obj, host_rel  # to store host data for graph plotting
    global para_gen, para_obj, para_rel  # to store para data for graph plotting

    host = hosts[len(hosts)//2] # gets the median host
    para = paras[len(hosts)//2] # gets the median parasite
    host["objective"] = host["solution"].count("1") # gets the objective fitness
    para["objective"] = para["solution"].count("1") # gets the objective fitness
    line1 = "{:4d} host: '{}' ({:.3f}) = {:3d}".format(gen, host["solution"], host["fitness"], host["objective"])
    line2 = "{:4d} para: '{}' ({:.3f}) = {:3d}".format(gen, para["solution"], para["fitness"], para["objective"])

    if file:
        file.write(line1+"\n")
        file.write(line2+"\n")
    else:
        print(line1)
        print(line2)

    for h in hosts:  # for every host
        host_gen.append(gen)  # adds the current generation to the gen list (will be used as x coord)
        host_obj.append(h["solution"].count("1"))  # adds the objective fitness (y coord)..
        host_rel.append(h["fitness"])  # ..and the relative fitness (z coord) to lists

    for p in paras:  # for every parasite
        para_gen.append(gen)  # adds the current generation to the gen list (will be used as x coord)
        para_obj.append(p["solution"].count("1"))  # adds the objective fitness (y coord)..
        para_rel.append(p["fitness"])  # ..and the relative fitness (z coord) to lists

# Defines the main function for the coevolutionary Genetic Algorithm (GA):
#   The function takes multiple parameters with default values to customize the evolutionary process.
#   Seeds the pseudo-random number generator for unique runs.
#   Initializes populations of individual "hosts" (solution designs) and "parasites" (solution challenges).
#   Assesses the initial populations.
#   Executes a specified number of generations (max_gen) of coevolution, with each generation involving:
#       Incrementing the generation counter.
#       Breeding new populations of offspring hosts and, depending on coevolution implementation, offspring parasites or generating new random parasites.
#       Applying mutations to both the new offspring hosts and parasites.
#       Assessing the fitness of each member in the new populations.
#       Optionally writing out statistics for the current generation.
#   Returns the final generation count and the best host and parasite from the last generation.

def do_the_ga(pop_size=50, length=100, tournament_size=5, max_gen=600, mutation_rate=0.03, para_lambda=1.0,
              num_competitions=10, alphabet="01", host_bias=[0.5,0.5], para_bias=[0.25,0.75], coevolution=True,
              write_every=10, file=None):

    random.seed()

    host_pop = initialise(pop_size, length, alphabet, zeropop = coevolution) # zeropop is a flag determining whether we initialise with 000.. bit strings or not
    para_pop = initialise(pop_size, length, alphabet, zeropop = coevolution) # zeropop is a flag determining whether we initialise with 000.. bit strings or not
    host_pop, para_pop = assess(host_pop, para_pop, num_competitions, para_lambda)
    generation = 0

    while generation < max_gen:

        generation += 1
        host_pop = breed(host_pop, tournament_size)

        if coevolution:
            para_pop = breed(para_pop, tournament_size)
        else:
            para_pop = initialise(pop_size, length, alphabet, zeropop = coevolution) # if not applying coevolution just produce a random bunch of parasites instead

        host_pop = mutate(host_pop, length, mutation_rate, alphabet, host_bias)
        para_pop = mutate(para_pop, length, mutation_rate, alphabet, para_bias)
        host_pop, para_pop = assess(host_pop, para_pop, num_competitions, para_lambda)

        if write_every and generation % write_every==0:
            write_fitness(host_pop, para_pop, generation, file)

    best_host = host_pop[0]
    best_para = para_pop[0]
    best_host["objective"] = best_host["solution"].count("1")
    best_para["objective"] = best_para["solution"].count("1")

    return generation, best_host, best_para

# Generates a scatter plot illustrating the coevolutionary dynamics between hosts and parasites:
#   Takes in the x (evolutionary generation), y (objective fitness), and z (relative fitness) coordinates for both hosts and parasites.
#   Plots objective fitness (y) against evolutionary generation (x), using relative fitness (z) to color the markers.
#   Utilizes distinct marker styles to differentiate between hosts (circle markers) and parasites (cross markers),
#       allowing for a clear visual comparison of their evolutionary trajectories over time.

def make_a_graph(x=[], y=[], z=[], series=[], write_every=10,
                 title="", xlabel="Axis Unlabelled!", ylabel="Axis Unlabelled!", zlabel="Axis Unlabelled!"):

    with plt.xkcd(): 

        # For optimal visual presentation of the scatter plot in matplotlib, it's recommended to use the Humor Sans MPL font:
        #   This font can be downloaded from https://seis.bristol.ac.uk/~sb15704/HumorSansMPL.ttf.
        #   Humor Sans MPL includes additional glyphs to address a font issue related to a matplotlib bug,
        #       enhancing the plot's readability and aesthetic appeal.
        # This step ensures that the scatter plot not only conveys the evolutionary dynamics effectively but does so with an engaging visual style.

        mpl.rcParams['font.family'] = ['Humor Sans MPL']   # uses the xkcd font with missing characters included
        mpl.rcParams['figure.figsize'] = [9.6, 7.2]        #    and sets the plot size to approx 10" by 7"

        # To address the limitation of matplotlib not automatically wrapping long titles, the following approach is used:
        #   The title string 'title' is split into a list of chunks, each with a maximum length of 30 letters.
        #   These chunks are then concatenated using newline characters to ensure the title spans multiple lines.
        # This method ensures that longer titles are displayed neatly across multiple lines, improving the readability and appearance of the plot.

        plt.title("\n".join(textwrap.wrap(title,32)))
        plt.xlabel(xlabel) # sets the x axis label
        plt.ylabel(ylabel) # sets the y axis label
        plt.ylim(bottom=0.0, top=100) # sets the yaxis to always range from 0 to 100
        # note: if the genotype is not length 100 the "top" value will have to be changed...

        cm = plt.cm.get_cmap('RdYlBu_r') # assigns a color map
        sc = [None, None]
        
        for s in range(len(y)):          # for each data set (first hosts and then parasites)
            marker = ['o', 'x'][s]       #  sets the scatter plot marker symbol
            # plot the markers:
            sc[s] = plt.scatter([pos+s*write_every/2 for pos in x[s]], y[s], c=z[s], marker=marker, cmap=cm, label=series[s], vmin=0, vmax=1)
            # note: shifts the parasite markers write_every/2 to the right to avoid them cluttering the host markers

        plt.legend()                            # adds the legend

        sc[0].set_alpha(0.25)                   # sets the markers to be somewhat transparent
        sc[1].set_alpha(0.25)                   # sets the markers to be somewhat transparent
        cbar = plt.colorbar(sc[0])              # adds a colour bar
        cbar.ax.set_ylabel(zlabel, rotation=90) # rotates the colour bar label
        cbar.solids.set(alpha=1)                # sets the colour bar colour to not be transparent

        plt.tight_layout()         # makes sure the plot has room for the axis labels and title
        plt.savefig(title+".pdf")  #    ..and saves it to a file
        plt.show()                 # puts the plot on the screen

def q3():

    global host_gen, host_obj, host_rel, para_gen, para_obj, para_rel

    print("3a")

    host_gen, host_obj, host_rel, para_gen, para_obj, para_rel = [], [], [], [], [], [] # initialises the data lists
    gens, best_host, best_para = do_the_ga(coevolution=False)                           # runs the genetic algorithm

    print("\nDone: Evolution")
    print("{:4d} host: '{}' ({:.3f}) = {:3d}".format(gens,best_host["solution"],best_host["fitness"], best_host["objective"]))
    print("{:4d} para: '{}' ({:.3f}) = {:3d}".format(gens,best_para["solution"],best_para["fitness"], best_para["objective"]))
    print("")
    print("Evolution is not under any pressure to improve the host solution once hosts are somewhat better than random bitstrings.\n")

    make_a_graph(x = [host_gen, para_gen], y = [host_obj, para_obj], z = [host_rel, para_rel],
                 title="Evolutionary GA Performance",
                 xlabel="Generation", ylabel="Objective Fitness", zlabel="Relative Fitness", series = ["Host", "Para"])

    input("")

    print("3b")

    host_gen, host_obj, host_rel, para_gen, para_obj, para_rel = [], [], [], [], [], []
    para_lambda = 1.0
    para_bias = 0.5
    gens, best_host, best_para = do_the_ga(coevolution=True)

    print("\nDone: Coevolution")
    print("{:4d} host: '{}' ({:.3f}) = {:3d}".format(gens,best_host["solution"],best_host["fitness"], best_host["objective"]))
    print("{:4d} para: '{}' ({:.3f}) = {:3d}".format(gens,best_para["solution"],best_para["fitness"], best_para["objective"]))
    print("")
    print("Coevolution took around 200 generations to find a perfect host.\n")

    make_a_graph(x = [host_gen, para_gen], y = [host_obj, para_obj], z = [host_rel, para_rel],
                 title="Coevolutionary GA Performance: parasite bias "+str(para_bias)+"; lambda = "+str(para_lambda),
                 xlabel="Generation", ylabel="Objective Fitness", zlabel="Relative Fitness", series = ["Host", "Para"])

    input("")

    print("3c")

    host_gen, host_obj, host_rel, para_gen, para_obj, para_rel = [], [], [], [], [], []
    para_lambda = 1.0
    para_bias = 0.9
    gens, best_host, best_para = do_the_ga(para_lambda=1, para_bias= [1-para_bias,para_bias])

    print("\nDone: Lamda=1.0 - Maximum Virulence; Strong parasite mutation bias [0.1, 0.9]")
    print("{:4d} host: '{}' ({:.3f}) = {:3d}".format(gens,best_host["solution"],best_host["fitness"], best_host["objective"]))
    print("{:4d} para: '{}' ({:.3f}) = {:3d}".format(gens,best_para["solution"],best_para["fitness"], best_para["objective"]))
    print("")
    print("With strong bias in favour of the parasites, hosts get left behind and the populations disengage.\n")
    print("Parasites drift to 90% 1s (their mutation bias), and hosts drift to their average genotype = ~50% 1s\n")

    make_a_graph(x = [host_gen, para_gen], y = [host_obj, para_obj], z = [host_rel, para_rel],
                 title="Coevolutionary GA Performance: parasite bias "+str(para_bias)+"; lambda = "+str(para_lambda),
                 xlabel="Generation", ylabel="Objective Fitness", zlabel="Relative Fitness", series = ["Host", "Para"])

    input("")

    print("3d")

    host_gen, host_obj, host_rel, para_gen, para_obj, para_rel = [], [], [], [], [], []
    para_lambda = 0.75
    para_bias = 0.9
    gens, best_host, best_para = do_the_ga(para_lambda=0.75, para_bias= [1-para_bias,para_bias])

    print("\nDone: Lamda=0.75 - Moderate Virulence; Strong parasite mutation bias [0.1, 0.9]")
    print("{:4d} host: '{}' ({:.3f}) = {:3d}".format(gens,best_host["solution"],best_host["fitness"], best_host["objective"]))
    print("{:4d} para: '{}' ({:.3f}) = {:3d}".format(gens,best_para["solution"],best_para["fitness"], best_para["objective"]))
    print("")
    print("Despite the strong bias in favour of the parasites, hosts no longer get left behind because of moderate virulence.\n")
    print("Parasites climb to around 97% 1s and so do hosts.\n")
    print("Why don't hosts and paras reach 100% 1s? Moderate virulence prevents parasites from really applying the final push.\n")

    make_a_graph(x = [host_gen, para_gen], y = [host_obj, para_obj], z = [host_rel, para_rel],
                 title="Coevolutionary GA Performance: parasite bias "+str(para_bias)+"; lambda = "+str(para_lambda),
                 xlabel="Generation", ylabel="Objective Fitness", zlabel="Relative Fitness", series = ["Host", "Para"])

    input("")

    print("3e")

    host_gen, host_obj, host_rel, para_gen, para_obj, para_rel = [], [], [], [], [], []
    para_lambda = 1.0
    para_bias = 0.6
    gens, best_host, best_para = do_the_ga(para_lambda=para_lambda, para_bias= [1-para_bias,para_bias])

    print("\nDone: Lamda=1.0 - Maximum Virulence; Weak parasite mutation bias [0.4, 0.6]")
    print("{:4d} host: '{}' ({:.3f}) = {:3d}".format(gens,best_host["solution"],best_host["fitness"], best_host["objective"]))
    print("{:4d} para: '{}' ({:.3f}) = {:3d}".format(gens,best_para["solution"],best_para["fitness"], best_para["objective"]))
    print("")
    print("Now with weaker bias in favour of the parasites, hosts no longer get left behind even with maximum virulence parasites.\n")
    print("Hosts are able to keep pace with parasites, and both climb to around 98% or 99% 1s.\n")
    print("Hosts and paras get closer to 100% 1s. Maximum virulence parasites are happy to punish hosts for missing even 1 or 2 '1's.\n")
    print("What if we reduced host mutation rate might that slow them down and allow disengagement to occur?\n")

    make_a_graph(x = [host_gen, para_gen], y = [host_obj, para_obj], z = [host_rel, para_rel],
                 title="Coevolutionary GA Performance: parasite bias "+str(para_bias)+"; lambda = "+str(para_lambda),
                 xlabel="Generation", ylabel="Objective Fitness", zlabel="Relative Fitness", series = ["Host", "Para"])

    input("")

    print("3f")

    host_gen, host_obj, host_rel, para_gen, para_obj, para_rel = [], [], [], [], [], []
    para_lambda = 0.75
    para_bias = 0.6
    gens, best_host, best_para = do_the_ga(para_lambda=para_lambda, para_bias= [1-para_bias,para_bias])
    
    print("\nDone: Lamda=0.75 - Moderate Virulence; Weak parasite mutation bias [0.4, 0.6]")
    print("{:4d} host: '{}' ({:.3f}) = {:3d}".format(gens,best_host["solution"],best_host["fitness"], best_host["objective"]))
    print("{:4d} para: '{}' ({:.3f}) = {:3d}".format(gens,best_para["solution"],best_para["fitness"], best_para["objective"]))
    print("")
    print("Staying with weaker bias in favour of the parasites, but employing moderate virulence parasites...\n")
    print("Hosts are able to keep pace with parasites, and both climb to around 93% or 95% 1s.\n")
    print("No problems with disengagement, but moderate virulence parasites are not interested in punishing hosts for missing the last few '1's.\n")

    make_a_graph(x = [host_gen, para_gen], y = [host_obj, para_obj], z = [host_rel, para_rel],
                 title="Coevolutionary GA Performance: parasite bias "+str(para_bias)+"; lambda = "+str(para_lambda),
                 xlabel="Generation", ylabel="Objective Fitness", zlabel="Relative Fitness", series = ["Host", "Para"])

q3()

