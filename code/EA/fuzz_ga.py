# fuzz_ga.py
import numpy as np
from ga import *
import matplotlib.pyplot as plt
from fit_funcs import *

f_funcs, domain = get_zdt()
pop_size = 24
dna_size = len(domain)
print(dna_size)
pm = 0.05

if_sampling = True
is_err_collection = 1

a = np.random.random_integers(0, 0, size=(6, 125))

np.savetxt("diversity.txt", a)

b = np.random.random_integers(1, 1, size=(6, 125))

np.savetxt("diversity_count.txt", b)

c = np.random.random_integers(1, 1, size=(6, 125))

np.savetxt("entropy.txt", c)

ga = ga(pop_size, dna_size, pc=1, pm=0.05, f_funcs=f_funcs, domain=domain)

P = ga.ini_pop(if_sampling)
print(len(P))

ga.run_sim(P, [False, is_err_collection,
               2])  # data_collection_para : [is_new_seed(for entropy), is_err_collection(collect err or not), err_type(collect normal(1)/sampling(2)/random data(3))]
ga.calculate_pop_fitness(P)

R = P

N = 25
for i in range(N):
    print(i)
    # Selection
    P, R = ga.select(R)  # P:selected_seed, R:all_seed
    # Cross
    Q = ga.pop_cross(P, i + 1)
    # Mutation
    Q = ga.pop_mutation(Q)
    # Simulation
    ga.run_sim(R, [False, 0, 0])
    ga.run_sim(Q, [True, is_err_collection, 1])
    # Merge
    R = R + Q
    # Cal Fitness
    ga.calculate_pop_fitness(R)
    ga.calulate_pop(R)
