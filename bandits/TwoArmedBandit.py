#####
# Two-armed bandit tests
#
#
#
#
#####
import numpy as np
import math
import matplotlib.pyplot as plt
import os

# @jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# @jit(nopython=True)
def project(x):
    # if x < 1e-12:
    #     return 1e-12
    # elif x > 1-1e-12:
    #     return 1-1e-12
    # else:
    #     return x
    return np.clip(x, 1e-12, 1-1e-12)

class Bandit:
    def __init__(self, r1, r2, init_param, noise=None, perturb_baseline=0.0, optimizer='natural',
                 parameterization='sigmoid', baseline_type="minvar", adaptive_base=False, entropy_reg=None, clip=None, seed=None):
        # This class defines both the environment and the agent
        self.r1 = r1
        self.r2 = r2
        self.noise = noise  # a tuple containing the standard deviation of the gaussian noises for each reward
        self.baseline_type = baseline_type  # "minvar", "value", None
        self.optimizer = optimizer  # in 'regular' (usual sgd), 'projected' (projected gd), 'natural' (natural gd)
        self.parameterization = parameterization  # in ('direct', 'sigmoid')

        self.init_param = init_param
        self.perturb_baseline = perturb_baseline  # how much to add to the optimal baseline

        self.param = init_param
        self.adaptive_base = adaptive_base
        self.entropy_reg = entropy_reg
        self.clip = clip  # the max update magnitude (before multiplying by step size)
        self.rng = np.random.RandomState(seed)

    def adaptive_baseline(self, test_param):
        p = self.get_prob(test_param)
        return min(p/(1-p), (1-p)/p)

    # @jit(nopython=True)
    def get_prob(self, test_param=None):
        ''' Returns prob of action 1'''
        if test_param is None:
            test_param = self.param

        if self.parameterization == 'direct':
            return test_param
        elif self.parameterization == 'sigmoid':
            return project(sigmoid(test_param))

    def get_optimal_baseline(self, test_param=None):
        ''' Returns the optimal baseline '''
        if test_param is None:
            test_param = self.param
        p = self.get_prob(test_param)
        return (1-p)*self.r1 + p*self.r2

    def get_sgd(self, test_param=None):
        ''' Returns stochastic gradient for current parameter '''
        p = self.get_prob(test_param)
        # print(p)
        if self.baseline_type == "minvar":
            if self.adaptive_base:
                b = self.get_optimal_baseline(test_param) + self.perturb_baseline*self.adaptive_baseline(test_param)
            else:
                b = self.get_optimal_baseline(test_param) + self.perturb_baseline
            # b = self.get_optimal_baseline(test_param) + self.perturb_baseline
        elif self.baseline_type == 'value':
            b = p * self.r1 + (1-p)*self.r2 + self.perturb_baseline  # the value function (expected reward)
        else:
            b = self.perturb_baseline

        rand = self.rng.uniform(0,1)
        update = None

        r1 = self.r1
        r2 = self.r2
        if self.noise is not None:
            r1 += np.random.normal(0, self.noise[0])
            r2 += np.random.normal(0, self.noise[1])

        if self.parameterization == 'direct':
            if rand < p:  # choose arm 1
                update = (r1-b) / p
            else:
                update = -(r2-b) / (1 - p)

            if self.entropy_reg is not None:
                update -= self.entropy_reg * np.log(p / (1-p))

        elif self.parameterization == 'sigmoid':
            if rand < p:  # choose arm 1
                update = (r1-b) * (1-p)
            else:
                update = -(r2-b) * p

            if self.entropy_reg is not None:
                update -= self.entropy_reg * p*(1-p)*np.log(p / (1-p))

        return update

    def do_sgd_step(self, step_size, test_param=None):
        ''' Performs an sgd step on the parameter '''
        if self.optimizer == 'projected':
            update = self.get_sgd()
            if self.clip is not None:
                update = np.clip(update, -self.clip, self.clip)
            self.param = project(self.param + step_size * update)
        elif self.optimizer == 'regular':
            update = self.get_sgd()
            if self.clip is not None:
                update = np.clip(update, -self.clip, self.clip)
            self.param += step_size * update
        elif self.optimizer == 'natural':
            p = self.get_prob(test_param)
            if self.parameterization == 'direct':
                update = p*(1-p)*self.get_sgd()
                if self.clip is not None:
                    update = np.clip(update, -self.clip, self.clip)
                self.param += step_size * update
            elif self.parameterization == 'sigmoid':
                update = self.get_sgd() / (p*(1-p))
                if self.clip is not None:
                    update = np.clip(update, -self.clip, self.clip)
                self.param += step_size * update

    def get_possible_gradients(self, test_param, return_next_params=False, step_size=0):
        ''' Returns the possible gradients and their probabilities starting at test_param
        The returned list is for [(grad1, prob1), (grad2, prob2)]
        test_param : the parameter at which to compute gradients
        return_next_params: if true, return the next parameter values instead of the gradients
        alpha: step size to use (only works if return_next_params is true)
        Note this doesn't work with noisy rewards '''

        p = self.get_prob(test_param)
        if self.baseline_type == "minvar":
            if self.adaptive_base:
                b = self.get_optimal_baseline(test_param) + self.perturb_baseline*self.adaptive_baseline(test_param)
            else:
                b = self.get_optimal_baseline(test_param) + self.perturb_baseline
        elif self.baseline_type == 'value':
            b = p * self.r1 + (1-p)*self.r2 + self.perturb_baseline  # the value function (expected reward)
        else:
            b = self.perturb_baseline


        if self.parameterization == 'direct':
            entropy = 0
            if self.entropy_reg is not None:
                entropy = -self.entropy_reg * np.log(p / (1-p))
            gradients = [((self.r1 - b) / p + entropy, p), (-(self.r2 - b) / (1 - p) + entropy, 1-p)]


        elif self.parameterization == 'sigmoid':
            entropy = 0
            if self.entropy_reg is not None:
                entropy = -self.entropy_reg * p*(1-p)*np.log(p / (1-p))
            gradients = [((self.r1 - b) * (1 - p) + entropy, p), (-(self.r2 - b) * p + entropy, 1-p)]

        if return_next_params:
            next_params = []
            for grad, prob in gradients:

                if self.optimizer == 'projected':
                    if self.clip is not None:
                        grad = np.clip(grad, -self.clip, self.clip)
                    next_params.append((project(test_param + step_size * grad), prob))
                elif self.optimizer == 'regular':
                    if self.clip is not None:
                        grad = np.clip(grad, -self.clip, self.clip)
                    next_params.append((test_param + step_size * grad, prob))

                if self.optimizer == 'natural':
                    if self.parameterization == 'direct':
                        update = grad * (prob*(1-prob))
                        if self.clip is not None:
                            update = np.clip(update, -self.clip, self.clip)
                        next_params.append((test_param + step_size * update, prob))
                    elif self.parameterization == 'sigmoid':
                        update = grad / (prob*(1-prob))
                        if self.clip is not None:
                            update = np.clip(update, -self.clip, self.clip)
                        next_params.append((test_param + step_size * update, prob))

            return next_params
        else:
            return gradients  # note doesn't clip these

    def do_sgd_step_action(self, step_size, action):
        ''' Does an update corresponding to the action specified.
        Returns the probability of the update occuring
        action: the index of the action taken '''
        updates = self.get_possible_gradients(self.param, True, step_size)

        # assert action in (1, 2)
        self.param = updates[action-1][0]
        return updates[action-1][1]

    def reset(self):
        self.param = self.init_param


def run_experiment(num_runs, num_steps, step_size, perturb, init_param, optimizer='natural', parameterization='sigmoid',
                   baseline_type="minvar", noise=None, entropy_reg=None, clip=None, zero_grad=False, adaptive_base=True, save_file=None, save_vars=None):
    param_data = []
    for i_run in range(num_runs):
        r1 = 0 if zero_grad else 1
        bandit = Bandit(r1, 0, init_param, noise=noise, perturb_baseline=perturb, optimizer=optimizer, parameterization=parameterization,
                        adaptive_base=adaptive_base, entropy_reg=entropy_reg, clip=clip, baseline_type=baseline_type)
        # TODO change rewards here

        param_seq = []
        for i_step in range(num_steps):
            bandit.do_sgd_step(step_size)
            param_seq.append(bandit.param)

        param_data.append(param_seq)

    param_data = np.array(param_data)
    if save_file is not None:
        path = save_file
        if save_vars is not None:
            for var in save_vars:
                path += "{}{}".format(var, locals()[var])  # adds the variables (specified in save_vars) and their values to the filename
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.save(path, param_data)
    return param_data

if __name__ == "__main__":
    ## training loop
    num_runs = 300
    num_steps = 300
    step_size = 2.0
    perturb = -1
    init_param = -1.0
    baseline_type = None
    entropy_reg = None
    noise = (0.0, 0.0)
    optimizer = 'regular'
    parameterization = 'sigmoid'
    save_file = 'results/param_data'
    save_file=None
    clip = None

    param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                       step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                       optimizer=optimizer, parameterization=parameterization, entropy_reg=entropy_reg, noise=noise,
                       init_param=init_param,
                       adaptive_base=False, clip=clip,
                       save_file=save_file)

    # print(param_data[:, -1])
    # print(np.unique(param_data[:, -1]))
    print("final avg performance", np.mean(sigmoid(param_data[:,-1])))
    bad_threshold = 0.01
    print("final proportion of bad <{}".format(bad_threshold), np.mean(sigmoid(param_data[:,-1]) < bad_threshold))

    # plt.hist(np.log(np.abs(param_data[:, -1])),  bins=100)

    ## plot the learning curves
    plt.figure()
    # num_lines = 200
    # for i in np.random.randint(0, num_runs, num_lines):
    for i in range(num_runs):
        plt.plot(sigmoid(param_data[i, :]), color='b', alpha=0.08)
    plt.plot(np.mean(sigmoid(param_data), axis=0), color='black')
    plt.ylim(0, 1)
    plt.title("epsilon {}".format(perturb))
    plt.ylabel('Prob. of right')
    plt.xlabel("Steps")

    ## color-coded learning curves
    plt.figure()
    # num_lines = 200
    # for i in np.random.randint(0, num_runs, num_lines):
    for i in range(num_runs):
        plt.plot(sigmoid(param_data[i, :]), color='b', alpha=0.08)
    plt.plot(np.mean(sigmoid(param_data), axis=0), color='black')
    plt.ylim(0, 1)
    plt.title("epsilon {}".format(perturb))
    plt.ylabel('Prob. of right')
    plt.xlabel("Steps")



    ## plot histogram of ending values
    plt.figure()
    plt.hist(sigmoid(param_data[:, -1]),  bins=100)
    plt.ylim(0, num_runs)

    plt.show()








    ## compute regret
    # def compute_regret(param_data):
    #     prob_data = sigmoid(param_data)
    #     regret = 1 - prob_data
    #     regret = np.cumsum(regret, axis=1)
    #     return regret
    #
    # regret = compute_regret(param_data)
    #
    # plt.figure()
    # for i in range(num_runs):
    #     plt.loglog(regret[i, :], color='lightblue', alpha=0.5)
    # plt.loglog(np.mean(regret, axis=0), color='blue')



    ## checking jumps from high theta to low theta and vice versa
    # this is to check how the regret behaves fo r
    # def check_jumps(param_data):
    #     # counts the number of jumps from high theta to low theta and vice versa
    #     jump_threshold_size = 2
    #     step_size = 0.1
    #     jump_up = 0
    #     jump_down = 0
    #     for i in range(param_data.shape[0]-1):
    #         downs = np.sum(param_data[:,i+1] - param_data[:, i] < -jump_threshold_size)
    #         if downs != 0:
    #             print("down at", i)
    #         jump_down += downs
    #
    #         jump_up += np.sum(param_data[:, i+1] - param_data[:, i] > jump_threshold_size)
    #
    #     return jump_up, jump_down
    #
    # print("num jumps", check_jumps(param_data))

    # np.save("quick_bandit_test_{}".format(perturb), param_data)
    # # load regret
    # plt.figure()
    # colors2 = ['lightblue', 'lightgreen']
    # colors = ['b', 'g']
    # perturbs = [1.0, -1.0]
    #
    # for i_hyp in (0,1):
    #
    #     perturb = perturbs[i_hyp]
    #     param_data = np.load("quick_bandit_test_{}.npy".format(perturb))
    #     prob_data = sigmoid(param_data)
    #
    #     regret = 1 - prob_data
    #     regret = np.cumsum(regret, axis=1)
    #
    #     for i in range(num_runs):
    #         plt.plot(regret[i,:], color=colors2[i_hyp], alpha=0.08)
    #     plt.plot(np.mean(regret, axis=0), color=colors[i_hyp])
    # plt.ylim(0, 50)
    # plt.title("Regrets")


    # random walk test
    # rw_data = []
    # for i in range(num_runs):
    #     rw_run = [0.0]
    #     for t in range(num_steps):
    #         if np.random.rand() < 0.5:
    #             rw_run.append(rw_run[t] - 3*step_size)
    #         else:
    #             rw_run.append(rw_run[t] + 3*step_size)
    #
    #     rw_data.append(rw_run)
    # rw_data = np.array(rw_data)
    # plt.hist(np.log(np.abs(rw_data[:, -1])),  bins=100)

    # perturbs = np.arange(-1, 6.1, 0.5)
    # []
    # for perturb in perturbs:
    #     print('perturb', perturb)

    #
    #
    # ###### For plotting
    # import numpy as np
    # import matplotlib.pyplot as plt
    # from utils import *
    #
    # # step_size = 0.03
    # # perturb = 5.0
    #
    # save_file + "_init{}_step{}_pert{}_{}_{}_steps{}".format(init_param, step_size, perturb, optimizer,
    #                                                          parameterization, num_steps)
    #
    # title = "step{}_pert{}_nat_sigmoid".format(step_size, perturb)
    # param_data = np.load("results/param_data_step{}_pert{}_nat_sigmoid_2step.npy".format(step_size, perturb))
    # num_runs, num_steps = param_data.shape
    #
    #
    # plt.figure()
    # plt.hist(sigmoid(param_data[:, -1]), bins=100, range=[0, 1])
    # plt.ylim(0, num_runs)
    # plt.title(title)
    #
    # print('final mean', np.mean(sigmoid(param_data[:, -1])))
    #
    # plt.figure()
    # # num_lines = 200
    # # for i in np.random.randint(0, num_runs, num_lines):
    # for i in range(num_runs):
    #     plt.plot(sigmoid(param_data[i, :]), color='b', alpha=0.08)
    # plt.ylim(0, 1)
    # plt.title(title)
    #
    # plt.show()






