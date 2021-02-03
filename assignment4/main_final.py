import numpy as np
from algorithms import LinUCB, RegretBalancingElim
from utils import FiniteContLinearRep, ContBanditProblem, make_random_rep, make_newlinrep_reshaped
import matplotlib.pyplot as plt
import time

from joblib import Parallel, delayed

def execute_regbalelim(T, true_rep, reps, reg_val, noise_std, delta, num):
    algo = RegretBalancingElim(reps, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print("--- {} finished in {} seconds ---".format(num, np.round(time.time() - start_time,2)))
    reg = problem.exp_instant_regret.cumsum()
    return reg

def execute_linucb(T, true_rep, rep, reg_val, noise_std, delta, num):
    algo = LinUCB(rep, reg_val, noise_std, delta)
    problem = ContBanditProblem(true_rep, algo, noise_std)
    start_time = time.time()
    problem.run(T)
    print("--- {} finished in {} seconds ---".format(num, np.round(time.time() - start_time,2)))
    reg = problem.exp_instant_regret.cumsum()
    return reg

PARALLEL = True
NUM_CORES = 6
NRUNS = 1
nc, na, nd = 200, 20, 20
noise_std = 0.3
reg_val = 1
delta = 0.01
T = 10000

b = np.load('final_representation2.npz')
reps = []
for i in range(b['true_rep']+1):
    feat = b['reps_{}'.format(i)]
    param = b['reps_param_{}'.format(i)]
    reps.append(FiniteContLinearRep(feat, param))
linrep = reps[b['true_rep']]

print("Running algorithm RegretBalancingElim")
results = []
if PARALLEL:
    results = Parallel(n_jobs=NUM_CORES)(
        delayed(execute_regbalelim)(T, linrep, reps, reg_val, noise_std, delta, i) for i in range(NRUNS)
    )
else:
    for n in range(NRUNS):
        results.append(
            execute_regbalelim(T, linrep, reps, reg_val, noise_std, delta, n)
        )
regrets = []
for el in results:
    regrets.append(el.tolist())
regrets = np.array(regrets)
mean_regret = regrets.mean(axis=0)
std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
plt.plot(mean_regret, label="RegBalElim")
plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

print("Running algorithm LinUCB")
for nf, f in enumerate(reps):

    results = []
    if PARALLEL:
        results = Parallel(n_jobs=NUM_CORES)(
            delayed(execute_linucb)(T, linrep, f, reg_val, noise_std, delta, i) for i in range(NRUNS)
        )
    else:
        for n in range(NRUNS):
            results.append(
                execute_linucb(T, linrep, f, reg_val, noise_std, delta, n)
            )

    regrets = []
    for el in results:
        regrets.append(el.tolist())
    regrets = np.array(regrets)
    mean_regret = regrets.mean(axis=0)
    std_regret = regrets.std(axis=0) / np.sqrt(regrets.shape[0])
    plt.plot(mean_regret, label="LinUCB - f{}".format(nf+1))
    plt.fill_between(np.arange(T), mean_regret - 2*std_regret, mean_regret + 2*std_regret, alpha=0.1)

plt.legend()
plt.ylabel("Pseudo-Regret")
plt.xlabel("Iterations")
plt.title("Pseudo-regret of the optimal LinUCB vs RegBalAlg")
plt.savefig("image_30.png")
plt.show()
