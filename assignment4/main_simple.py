import numpy as np
import algorithms
from algorithms import LinUCB, RegretBalancingElim
from utils import FiniteContLinearRep, ContBanditProblem, make_random_rep, make_newlinrep_reshaped
import matplotlib.pyplot as plt
import time

results = []

nc, na, nd = 10, 5, 4
noise_std = 0.3
delta = 0.01
reg_val = 1
T = 10000

scale = 1
adaptive_ci = True

linrep = make_random_rep(nc, na, nd, False)
algo = LinUCB(linrep, reg_val, noise_std, delta)
problem = ContBanditProblem(linrep, algo, noise_std)
start_time = time.time()
problem.run(T)
print("--- finished in {} seconds ---".format(np.round(time.time() - start_time,2)))
reg = problem.exp_instant_regret.cumsum()

plt.plot(reg, label="LinUCB")

# # reps_nested = [make_newlinrep_reshaped(linrep, i) for i in [5, 10, 25]]
reps_random = [ make_random_rep(nc, na, i, False) for i in [nd+1] ]
reps = reps_random + [linrep]
algo = RegretBalancingElim(reps, reg_val, noise_std, delta)
problem = ContBanditProblem(linrep, algo, noise_std)
start_time = time.time()
problem.run(T)
print("--- finished in {} seconds ---".format(np.round(time.time() - start_time,2)))
reg = problem.exp_instant_regret.cumsum()
plt.plot(reg, label="RegBalElim")

linUCB on random REP
algo = LinUCB(reps_random[0], reg_val, noise_std, delta)
problem = ContBanditProblem(linrep, algo, noise_std)
start_time = time.time()
problem.run(T)
print("--- finished in {} seconds ---".format(np.round(time.time() - start_time,2)))
reg = problem.exp_instant_regret.cumsum()
plt.plot(reg, label="LinUCB-random")

plt.legend()

plt.show()
