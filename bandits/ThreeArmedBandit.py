#####
# Three armed bandit
#
#
#
#####
import numpy as np
from numba import jit
import math
import matplotlib.pyplot as plt
import time
import ternary

def plot_trajectories(trajectory_data, title, nocolor=False, savefile=None):

    num_runs = trajectory_data.shape[0]

    figure, tax = ternary.figure(scale=1.0)
    tax.boundary()
    tax.get_axes().axis('off')
    tax.gridlines(multiple=0.2, color="black")
    tax.set_title(title+'\n', fontsize=20)
    tax.right_corner_label("R = 1.0")
    tax.top_corner_label("R = 0.7")
    tax.left_corner_label("R = 0.0")
    for run in range(num_runs):
        points = trajectory_data[run]
        if nocolor:
            tax.plot(points, linewidth=1.0, alpha=0.5, color='black')
        else:
            tax.plot_colored_trajectory(points, linewidth=1.0, alpha=0.5)
        tax.scatter([points[0]], marker='o', color='black', s=64)
        tax.scatter([points[-1]], marker='o', color='red', s=64, edgecolors='red')
    # tax.show()
    return figure, tax

def plot_baseline_stats(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data = np.mean(data, axis=0)
    plt.plot(data)
    plt.plot(data_max, label='max')
    plt.plot(data_min, label='min')

# @jit(nopython=True)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

def escort(x, p=2):
    # note this is not good for x = 0
    x = np.power(np.abs(x), p)
    return x / np.sum(x)

def basis_vector(i):
    vec = np.zeros(3, dtype='float')
    vec[i] = 1
    return vec


# @jit(nopython=True)
def project(x):
    return np.clip(x, 1e-8, 1-1e-8)

class ThreeArmedBandit:
    def __init__(self, rewards, init_param, baseline_type='minvar', perturb_baseline=0.0, optimizer='vanilla', parameterization='softmax', seed=None, **kwargs):
        # This class defines both the environment and the agent
        self.rewards = rewards  # list of rewards
        self.baseline_type = baseline_type #  either 'minvar' or 'value'
        self.optimizer = optimizer  # in 'vanilla' (usual sgd), 'projected' (projected gd), 'natural' (natural gd)
        self.parameterization = parameterization  # in ('direct', 'softmax')  # only softmax works for now

        if parameterization == 'escort':
            # self.excort_p = escort
            self.escort_p = 2.0

        self.init_param = init_param  # list of initial parameters
        self.perturb_baseline = perturb_baseline  # how much to add to the baseline

        self.param = init_param
        self.rng = np.random.RandomState(seed)

    # @jit(nopython=True)
    def get_prob(self, test_param=None):
        ''' Returns prob of action 1'''
        if test_param is None:
            test_param = self.param

        # if self.parameterization == 'direct':
        #     return test_param
        # print(self.parameterization)
        if self.parameterization == 'softmax':
            p = project(softmax(np.array(test_param)))
            # print('getprob', test_param, p)

            return p / np.sum(p)  # renormalize in case of clipping
        elif self.parameterization == 'escort':
            p = escort(np.array(test_param))
            p = project(p)
            return p / np.sum(p)

    def get_optimal_baseline(self, test_param=None):
        ''' Returns the optimal baseline '''
        if test_param is None:
            test_param = self.param

        p = self.get_prob(test_param)
        baseline = np.sum([self._weight_optimal_baseline(i, p, test_param) * self.rewards[i] for i in range(3)])

        return baseline

    def _weight_optimal_baseline(self, index, policy_probs, param):
        # p_i || \log \pi(i) ||^2_2 / E [ || \log \pi(i) ||^2_2 ]
        if self.parameterization == 'softmax':
            if self.optimizer == 'vanilla':
                # policy_probs = np.clip(policy_probs, 1e-6, 1)
                denom = np.sum([policy_probs[index] / policy_probs])
                weight = 1 / denom
            elif self.optimizer == 'natural':
                denom = 1 - np.sum(np.square(policy_probs))
                num = np.sum(np.square(policy_probs)) - policy_probs[index]**2 + (1-policy_probs[index])**2
                weight = num * policy_probs[index] / denom
        elif self.parameterization == 'escort':
            if self.optimizer == 'vanilla':
                p = self.escort_p
                onehot = np.zeros(3)
                onehot[index] = 1
                prob_logprobnorm = p**2 / np.linalg.norm(param)**2 * \
                                   np.sum(np.square(np.power(policy_probs, -1 / p) * (onehot - policy_probs)))
                weight = prob_logprobnorm[index] / np.sum(prob_logprobnorm)
        return weight

    def get_sgd(self, test_param=None):
        ''' Returns stochastic gradient for current parameter '''
        pol = self.get_prob(test_param)

        if self.baseline_type == 'minvar':
            b = self.get_optimal_baseline(test_param) + self.perturb_baseline
            # b = self.get_optimal_baseline(test_param) + self.perturb_baseline
        elif self.baseline_type == 'value':
            b = np.sum(np.array(self.rewards)*pol) + self.perturb_baseline
        else:
            b = self.perturb_baseline

        # act = self.rng.randint(0, 3)
        act = self.rng.choice([0,1,2], p=pol)
        # print("act", act)
        if self.parameterization == 'softmax':
            if self.optimizer == 'vanilla':
                onehot = np.zeros(3)
                onehot[act] = 1
                grad = onehot - pol

            if self.optimizer == 'natural':
                # this is the minimum-norm update
                # grad = np.ones(3, dtype='float') / (2*p[act]) + 1 / p[act] * basis_vector(act)
                grad = -np.ones(3, dtype='float') / (3 * pol[act]) + 1 / pol[act] * basis_vector(act)
        elif self.parameterization == 'escort':
            if self.optimizer == 'vanilla':
                onehot = np.zeros(3)
                onehot[act] = 1
                p = self.escort_p
                grad = np.power(pol, -1/p) * np.sign(self.param) * (onehot - pol)
                grad = grad * p / np.linalg.norm(self.param, ord=p)
        else:
            raise AssertionError('invalid parameterization or optimizer')

        return (self.rewards[act] - b) * grad


    def do_sgd_step(self, step_size, test_param=None):
        ''' Performs an sgd step on the parameter '''
        self.param = self.param + step_size * self.get_sgd(test_param)


    def reset(self):
        self.param = self.init_param


def run_experiment(num_runs, num_steps, rewards, step_size, perturb, init_param, baseline_type, optimizer, parameterization='softmax', save_file=None):
    param_data = []
    prob_data = []
    for i_run in range(num_runs):
        bandit = ThreeArmedBandit(rewards, init_param, baseline_type, perturb, optimizer, parameterization)


        param_seq = []
        prob_seq = []
        for i_step in range(num_steps):
            bandit.do_sgd_step(step_size)
            param_seq.append(bandit.param)
            prob_seq.append(bandit.get_prob())

        param_data.append(param_seq)
        prob_data.append(prob_seq)

    param_data = np.array(param_data)
    prob_data = np.array(prob_data)
    if save_file is not None:
        np.save(save_file + "_init{}_step{}_pert{}_{}_{}_steps{}".format(init_param, step_size, perturb, optimizer, parameterization, num_steps), param_data)
    return param_data, prob_data

if __name__ == "__main__":
    ## training loop
    rewards = [1, 0.7, 0]
    num_runs = 15
    num_steps = 1000
    step_size = 0.5
    perturb = -0.5
    init_param = np.array([1.0, 1.0, 1.0])  # [0, 0.2, 2] [0, 2,5]
    # print(softmax(init_param))
    optimizer = 'vanilla'
    parameterization = 'escort'
    baseline_type = 'minvar'
    save_file = None #'results/param_data'

    param_data, prob_data = run_experiment(num_runs=num_runs, num_steps=num_steps, rewards=rewards,
                       step_size=step_size, perturb=perturb, baseline_type=baseline_type, optimizer=optimizer,
                       init_param=init_param,
                       save_file=save_file)

    plt.figure()
    # num_lines = 200
    for i in range(num_runs):
        plt.plot((prob_data[i, :, 0]), color='b', alpha=0.2)
        # plt.plot((prob_data[i, :, 1]), color='g', alpha=0.2)
        # plt.plot((prob_data[i, :, 2]), color='r', alpha=0.2)
    plt.ylim(0, 1)
    plt.ylabel("prob of act 1")

    # plt.figure()
    # num_lines = 200
    # for i in range(num_runs):
    #     plt.plot((prob_data[i, :, 1]), color='g', alpha=0.2)
    # plt.ylim(0, 1)
    # plt.ylabel("prob of act 2")

    plt.figure()
    avg_rewards = []
    for i in range(num_runs):
        avg_reward_traj = np.sum(prob_data[i, :, :] * np.array(rewards), axis=1)
        avg_rewards.append(avg_reward_traj)
        plt.plot(avg_reward_traj, color='b', alpha=0.2)
    avg_rewards = np.array(avg_rewards)

    mean = np.mean(avg_rewards, axis=0)
    std = np.std(avg_rewards, axis=0)
    plt.plot(np.mean(avg_rewards, axis=0), color='red', linewidth=2)
    plt.fill_between(range(len(mean)), mean - std, mean + std, alpha=0.10, color='red')
    plt.ylim(0,1)
    plt.ylabel("average reward")

    # bad_threshold = 0.01
    # print("final proportion of bad <{}".format(bad_threshold), np.mean(prob_data[:, -1, 0] < bad_threshold))

    plot_trajectories(prob_data[:, :, :], "baseline{}".format(perturb))




    # for producing gif
    def save_ternary_gif(prob_data, plot_name, num_frames=100, folder_name=''):
        # prob_data dimensions are num_runs x num_steps x num_actions

        os.makedirs('three_armed_bandit_gif/' + subfolder + plot_name, exist_ok=True)

        N = prob_data.shape[1]
        for i in range(0, N+1, int(N/num_frames)):
            fig, tax = plot_trajectories(prob_data[:, 0:(max(1,i)), :], plot_name+' step '+str(i), nocolor=True)
            fig.savefig('three_armed_bandit_gif/{}{}/{}_frame_{}.png'.format(folder_name, plot_name, plot_name, i), bbox_inches='tight')
            # plt.clf()
            # plt.close()
            # time.sleep(5)
            plt.clf()

            plt.close()

    import os
    subfolder = 'vpg'

    # os.makedirs('three_armed_bandit_gif/' + subfolder, exist_ok=True)

    save_ternary_gif(prob_data, "baseline{}".format(perturb), folder_name=subfolder)

    quit()
    #
    # import glob
    # from PIL import Image
    #
    # fp_in = "/path/to/image_*.png"
    # fp_out = "/path/to/image.gif"
    #
    # # https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
    # img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    # img.save(fp=fp_out, format='GIF', append_images=imgs,
    #          save_all=True, duration=200, loop=0)



    # print(param_data[:, -1])
    # print(np.unique(param_data[:, -1]))
    print("final avg performance", np.mean(sigmoid(param_data[:,-1])))
    bad_threshold = 0.01
    print("final proportion of bad <{}".format(bad_threshold), np.mean(sigmoid(param_data[:,-1]) < bad_threshold))

    # plt.figure()
    # plt.hist(sigmoid(param_data[:, -1]),  bins=100)
    # plt.hist(np.log(np.abs(param_data[:, -1])),  bins=100)

    # plot the learning curves
    plt.figure()
    # num_lines = 200
    # for i in np.random.randint(0, num_runs, num_lines):
    for i in range(num_runs):
        plt.plot(sigmoid(param_data[i, :]), color='b', alpha=0.08)
    plt.ylim(0, 1)

    # compute regret
    def compute_regret(param_data):
        prob_data = sigmoid(param_data)
        regret = 1 - prob_data
        regret = np.cumsum(regret, axis=1)
        return regret

    regret = compute_regret(param_data)

    plt.figure()
    for i in range(num_runs):
        plt.loglog(regret[i, :], color='lightblue', alpha=0.5)
    plt.loglog(np.mean(regret, axis=0), color='blue')



