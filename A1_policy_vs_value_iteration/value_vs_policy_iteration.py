import sys
sys.path.insert(0, './utils')
import numpy as np
import matplotlib.pyplot as plt
import math
from cliffwalk import CliffWalk
import time
import sys


def policy_evaluation(P, R, policy, gamma=0.9, tol=1e-2):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        policy: np.array
            matrix mapping states to action (Ns)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        value_function: np.array
            The value function of the given policy
    """
    Ns, Na = R.shape

    P_pi=[] #policy transition matrix
    R_pi=[] #policy reward vector
    i=0

    #Computing the policy reward vector and policy transition matrix
    for a in policy:
        P_pi.append(P[i,a,:])
        R_pi.append(R[i,a])
        i+=1

    P_pi=np.array([P_pi]).reshape((48,48))
    R_pi=np.array([R_pi])

    #Computing the value function of the associated to the policy, with the direct formula
    #value_function=np.dot(np.linalg.inv(np.eye(Ns)-gamma*P_pi),R_pi.T)
    value_function=np.linalg.solve(np.eye(Ns)-gamma*P_pi,R_pi.T)

    return value_function

def policy_iteration(P, R, gamma=0.9, tol=1e-3):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        policy: np.array
            the final policy
        V: np.array
            the value function associated to the final policy
    """
    Ns, Na = R.shape
    V = np.zeros(Ns)
    policy = np.zeros(Ns, dtype=np.int)

    #Initialising a second policy different from the first to start the while loop
    previous_policy=np.ones(Ns,dtype=np.int)

    k=0
    #As long as the new policy is different from the previous one, we continue policy iteration
    while np.any(policy!=previous_policy):
        previous_policy=policy.copy()

        #Evaluation of the current policy
        V=policy_evaluation(P,R,policy)

        # #Improve policy for each state using the greedy policy
        X=R+gamma*np.dot(P,V).reshape((48,4))
        policy=X.argmax(axis=1)

        k+=1
    print("Number of policy iterations before convergence with policy iteration:",k)

    return policy, V

def value_iteration(P, R, gamma=0.9, tol=1e-3):
    """
    Args:
        P: np.array
            transition matrix (NsxNaxNs)
        R: np.array
            reward matrix (NsxNa)
        gamma: float
            discount factor
        tol: float
            precision of the solution
    Return:
        Q: final Q-function (at iteration n)
        greedy_policy: greedy policy wrt Qn
        Qfs: all Q-functions generated by the algorithm (for visualization)
    """
    Ns, Na = R.shape
    Q = np.zeros((Ns, Na))
    Qfs = [Q]

    #Initialising while loop condition
    M=1
    k=0

    #Until the termination condition is met:
    while M>tol:
        Q_old=Q.copy()
        #Compute the new Q function using the Bellman operator equality
        Q=R+gamma*np.dot(P,Q.max(axis=1))
        Qfs.append(Q)

        #M=np.linalg.norm(Q-Q_old,ord="inf")
        M=max(np.sum(abs(Q-Q_old),axis=1))
        k+=1

    print("Number of policy iterations before convergence with value iteration:",k)

    #Return the greedy policy for the last Q
    greedy_policy=Q.argmax(axis=1)

    return Q, greedy_policy, Qfs

# Edit below to run policy and value iteration on different environments and
# visualize the resulting policies in action!
if __name__ == "__main__":
    tol =1e-5
    env = CliffWalk(proba_succ=1)
    #print(env.R.shape)
    #print(env.P.shape)
    #env.render()

    #### POLICY ITERATION ####
    PI_policy, PI_V = policy_iteration(env.P, env.R, gamma=env.gamma, tol=tol)
    print("\n[PI]final policy:")
    env.render_policy(PI_policy)

    # run value iteration to obtain Q-values
    VI_Q, VI_greedypol, all_qfunctions = value_iteration(env.P, env.R, gamma=env.gamma, tol=tol)

    # render the policy
    print("[VI]Greedy policy: ")
    env.render_policy(VI_greedypol)

    # compute the value function of the greedy policy using matrix inversion
    greedy_V = np.zeros((env.Ns, env.Na))

    # compute value function of the greedy policy
    VI_greedypol=VI_greedypol.astype(np.int)
    greedy_V=policy_evaluation(env.P, env.R, VI_greedypol, gamma=0.9, tol=1e-2)

    norms = [ np.linalg.norm(q.max(axis=1) - VI_Q.max(axis=1)) for q in all_qfunctions]
    plt.plot(norms)
    plt.xlabel('Iteration')
    plt.ylabel('Error')
    plt.title("Value iteration: convergence")

    # control that everything is correct
    assert np.allclose(PI_policy, VI_greedypol),\
        "You should check the code, the greedy policy computed by VI is not equal to the solution of PI"
    assert np.allclose(PI_V, greedy_V),\
        "Since the policies are equal, even the value function should be"

    # for visualizing the execution of a policy, use the following code
    # state = env.reset()
    # env.render()
    # for i in range(15):
    #     action = VI_greedypol[state]
    #     state, reward, done, _ = env.step(action)
    #     env.render()


    plt.show()