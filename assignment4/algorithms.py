#Pierre Ebert
import numpy as np
import math
from math import log, sqrt



class LinUCB:

    def __init__(self,
        representation,
        reg_val, noise_std, delta=0.01
    ):
        self.representation = representation
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = representation.param_bound
        self.features_bound = representation.features_bound
        self.delta = delta
        self.reset()

    def reset(self):
        ### TODO: initialize necessary info
        self.dim = self.representation.dim()
        self.inv_A = np.eye(self.dim) #Initial matrix A: identity matrix
        self.b = np.zeros(self.dim ) #Initial vector b: zeros
        ###################################
        self.t = 1

    def sample_action(self, context):
        ### TODO: implement action selection strategy
        #Calculation of theta
        theta = np.dot(self.inv_A, self.b)
        #Context is chosen at random for us (in utils.py) -> we only retrieve the selected context
        phi = self.representation.features[context]

        #Calculation of mu and alpha
        mu = np.dot(phi, theta)
        alpha = self.noise_std * np.sqrt(self.dim *np.log((1+self.t*self.features_bound**2/self.reg_val)/self.delta))+np.sqrt(self.reg_val)*self.param_bound

        #Calculation of the interval
        I = np.sqrt(np.diag(np.dot(np.dot(phi,self.inv_A),phi.T)))

        #Calculation of the upper_bound for each action
        B = mu + alpha * I

        #Select best action
        action = np.argmax(B)
        ###################################
        self.t += 1
        return action

    def update(self, context, action, reward):
        v = self.representation.get_features(context, action)
        ### TODO: update internal info (return nothing)
        #Updating b and A according to the relevant formulas
        self.b = self.b + v*reward
        #Sherman Morisson Formula to avoid using np.linalg.inv (more costly)
        self.inv_A = self.inv_A - np.outer(np.dot(self.inv_A, v), np.dot(v.T, self.inv_A)) / (1+np.dot(np.dot(v.T,self.inv_A),v))
        ###################################


class RegretBalancingElim:
    def __init__(self,
        representations,
        reg_val, noise_std,delta=0.01
    ):
        self.representations = representations
        self.reg_val = reg_val
        self.noise_std = noise_std
        self.param_bound = [r.param_bound for r in representations]
        self.features_bound = [r.features_bound for r in representations]
        self.delta = delta
        self.last_selected_rep = None
        self.active_reps = None # list of active (non-eliminated) representations
        self.t = None
        self.reset()


    def reset(self):
        ### TODO: initialize necessary info
        self.n = len(self.representations) #Number of learners at the beginning
        self.active_set = [i for i in range(self.n)] #Active set of learners
        self.rep_visits = np.zeros(self.n) #Number of times each learner is played
        self.cum_rewards = np.zeros(self.n) #Rewards accumulated by each learner
        self.dim = self.representations[0].dim() #Dimension of the representations
        self.param_bound = np.array(self.param_bound) #Bound the parameter norm
        self.features_bound = np.array(self.features_bound) #Bound on the action norm
        self.learner_bounds = np.array(self.n) #Pseudo regret bound
        self.M = len(self.active_set) #Number of active learners
        self.betas = [] #Betas collected at each rounds

        #Initialisation of the LinUCB with the representations provided
        self.algos = []
        for rep in self.representations:
            self.algos.append(LinUCB(rep, self.reg_val, self.noise_std, self.delta))
        ###################################
        self.t = 1

    def optimistic_action(self, rep_idx, context):
        ### TODO: implement action selection strategy given the selected representation
        #Selecting the best learner
        learner = self.active_set[rep_idx]
        self.last_selected_rep = learner

        #Playing the best learner
        self.algo = self.algos[learner]
        action = self.algo.sample_action(context)
        ###################################
        return maxa

    def sample_action(self, context):
        ### TODO: implement representation selection strategy and action selection strategy
        #Calculating beta_t for all learners
        beta = self.noise_std * np.sqrt(self.dim *np.log((1+self.rep_visits*self.features_bound**2/self.reg_val)/self.delta))+np.sqrt(self.reg_val)*self.param_bound

        self.betas.append(beta)

        #Obtaining beta_max (maximum of t)
        beta = np.max(self.betas)

        #Initialisation to avoid ln(0) and /0 errors
        if self.t <= self.n:
            self.learner_bounds = 2*beta*np.sqrt(self.dim*self.rep_visits*(1+self.features_bound**2/self.reg_val)*1) #to avoid ln(0)

        #Compute pseudo-regret for each learner
        else:
            self.learner_bounds = 2*beta*np.sqrt(self.dim*self.rep_visits*(1+self.features_bound**2/self.reg_val)*np.log((self.dim*self.reg_val+self.rep_visits*self.features_bound)/(self.dim*self.reg_val)))

        #Selecting the learner with the lowest pseudo regret
        bounds = self.learner_bounds[self.active_set]
        learner = np.argmin(bounds)
        learner = self.active_set[learner]
        self.last_selected_rep = learner

        #Playing the learner
        self.algo = self.algos[learner]
        action = self.algo.sample_action(context)

        ###################################
        self.t += 1
        return action

    def update(self, context, action, reward):
        idx = self.last_selected_rep
        v = self.representations[idx].get_features(context, action)
        ### TODO: implement update of internal info and active set

        #Updating the learner
        self.algo.update(context, action, reward)

        #Updating learner and visits stats
        self.rep_visits[idx] +=1
        self.cum_rewards[idx] +=reward

        #Perform misspecification test on each learner in the active set
        for idx in self.active_set:
            #Constant
            c = 2*0.85

            #Initialisation rounds: no elminination
            if self.t <= self.n or (self.rep_visits <=1).any() :
                pass
            #Perform misspecification testing
            else:
                #Calculating the upper bound of the tested learner
                upper_bound = self.cum_rewards[idx]/self.rep_visits[idx] + self.features_bound[idx]/self.rep_visits[idx] + c * np.sqrt((np.log(self.M*np.log(self.rep_visits[idx])/self.delta))/self.rep_visits[idx])

                #Calculating the max of the lower bounds of the other learners
                lower_bound = np.max(self.cum_rewards/self.rep_visits - c * np.sqrt((np.log(self.M*np.log(self.rep_visits)/self.delta))/self.rep_visits))

                #Performing the test: if UB < max(LB), the learner is removed from the active set
                if upper_bound < lower_bound and len(self.active_set)>1:
                    self.active_set.remove(idx)
                    break

        print("Active set", np.array(self.active_set)+1)
        ###################################
