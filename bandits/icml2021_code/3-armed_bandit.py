
'''
This code runs the 3-armed bandit experiment and generates plots.

Requires the installation of ternary package: pip install python-ternary

Example usage:
$python 3-armed_bandit.py minvar 0.5
for running the script with the minimum variance baseline and a perturbation of +0.5
More generally
$python 3-armed_bandit.py arg1 arg2
where arg1 is one of ['minvar', 'value', 'fixed']
and arg2 is a real number

The other arguments (save files, parameterization, number of steps, learning
rate, initial parameters) are to be changed manually in the main function
By default, these are set to match the simplex plots in the paper's main text:
softmax parameterization, natural policy gradient, initial theta of [0, 3, 5], step size of 0.025
Use arguments "minvar -0.5", "minvar 0", "minvar 0.5" and "value 0" to reproduce those plots.
'''


import numpy as np
import sys
import matplotlib.pyplot as plt
import ternary  # package to install is python-ternary
import os

# All the parameters
rewards = np.array([1.0, 0.7, 0])
num_runs = 25
num_steps = 1000
step_size = 0.025
baseline_type = str(sys.argv[1])  # 'value', 'minvar' or 'constant'
perturb = float(sys.argv[2])
optimizer = 'natural'  # 'vanilla' or 'natural'
parameterization = 'softmax'  # 'softmax' or 'direct' (direct only works with vanilla optimizer)
init_param = np.array([0.0, 3.0, 5.0])  # this is the same initialization used in the main text


def plot_trajectories(trajectory_data, title=None, theta_0=None, step_size=None):
    num_runs = trajectory_data.shape[0]

    figure, tax = ternary.figure(scale=1.0)
    tax.boundary(linewidth=2, alpha=0.3)
    tax.right_corner_label("R = 1.0", fontsize=15)
    tax.top_corner_label("R = 0.7", fontsize=15)
    tax.left_corner_label("R = 0.0",fontsize=15)
    
    ## Uncomment this block to plot the true gradient trajectory
    ## (only for NPG with softmax parameterization)
    #if theta_0 is not None:
    #    L = trajectory_data.shape[1]
    #    true_point = []
    #    theta_i = theta_0
    #    for i in range(L):
    #        true_point.append(softmax(theta_i))
    #        theta_i += step_size*np.array([1, 0.7, 0])
    #    tax.plot(true_point, linewidth=4.0, alpha=0.6, color='black')

    for run in range(num_runs):
        points = trajectory_data[run]
        tax.plot_colored_trajectory(points, linewidth=2.0, alpha=0.90)
    for run in range(num_runs):
        points = trajectory_data[run]    
        #tax.scatter([points[-1]], marker='o', color='black', s=64, edgecolors='red')

    tax.scatter([points[0]], marker='o', color='black', s=64)
    ax = tax.get_axes()
    ax.axis('off')
    # ax.set_aspect(2/np.sqrt(3))  # used to make the plots less wide for the main text
    tax.set_title(title, size=13, y=1.08)

    tax.show()
    return figure, tax

def plot_baseline_stats(data):
    data_max = np.max(data, axis=0)
    data_min = np.min(data, axis=0)
    data = np.mean(data, axis=0)
    plt.plot(data)
    plt.plot(data_max, label='max')
    plt.plot(data_min, label='min')

def softmax(x):
    x = x - np.max(x)
    x = np.exp(x)
    return x / np.sum(x)

def basis_vector(i):
    vec = np.zeros(3, dtype='float')
    vec[i] = 1
    return vec



def project(v, z=1):
    ''' using code from
    Large-scale Multiclass Support Vector Machine Training via Euclidean Projection onto the Simplex
    Mathieu Blondel, Akinori Fujino, and Naonori Ueda.
    ICPR 2014.
    http://www.mblondel.org/publications/mblondel-icpr2014.pdf
    '''
    if np.any(np.isnan(v)): print(v)
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

class ThreeArmedBandit:
    def __init__(self, rewards, init_param, baseline_type='minvar', perturb_baseline=0.0, optimizer='vanilla', parameterization='softmax', seed=None):
        # This class defines both the environment and the agent
        self.rewards = rewards  # list of rewards
        self.baseline_type = baseline_type #  either 'minvar' or 'value'
        self.optimizer = optimizer  # in 'vanilla' (usual sgd), 'natural' (natural gd)
        self.parameterization = parameterization  # in ('direct', 'softmax')  # only softmax works for now

        self.init_param = init_param  # list of initial parameters
        self.perturb_baseline = perturb_baseline  # how much to add to the baseline

        self.param = init_param
        self.rng = np.random.RandomState(seed)

    # @jit(nopython=True)
    def get_prob(self, test_param=None):
        ''' Returns prob of action 1'''
        if test_param is None:
            test_param = self.param

        if self.parameterization == 'direct':
            return project(np.array(test_param))
        elif self.parameterization == 'softmax':
            clip = 1e-8  # clipping is used for numerical stability and avoiding division by ~0
            p = np.clip(softmax(np.array(test_param)), clip, 1-clip)
            p /= np.sum(p)
            return p


    def get_optimal_baseline(self, test_param=None):
        ''' Returns the optimal baseline '''
        if test_param is None:
            test_param = self.param

        p = self.get_prob(test_param)
        baseline = np.sum([self._weight_optimal_baseline(i, p) * self.rewards[i] for i in range(3)])

        return baseline

    def _weight_optimal_baseline(self, index, policy_probs):
        if self.parameterization == 'softmax':
            if self.optimizer == 'vanilla':
                # policy_probs = np.clip(policy_probs, 1e-6, 1)
                denom = np.sum([policy_probs[index] / policy_probs])
                weight = 1 / denom
            elif self.optimizer == 'natural':
                denom = 1 - np.sum(np.square(policy_probs))
                num = np.sum(np.square(policy_probs)) - policy_probs[index]**2 + (1-policy_probs[index])**2
                weight = num * policy_probs[index] / denom
        elif self.parameterization == 'direct':
            weight = self.param

        return weight

    def get_sgd(self, test_param=None):
        ''' Returns stochastic gradient for current parameter '''
        p = self.get_prob(test_param)

        if self.baseline_type == 'minvar':
            b = self.get_optimal_baseline(test_param) + self.perturb_baseline
            # b = self.get_optimal_baseline(test_param) + self.perturb_baseline
        elif self.baseline_type == 'value':
            b = np.sum(np.array(self.rewards)*p) + self.perturb_baseline
        else:
            b = self.perturb_baseline

        # act = self.rng.randint(0, 3)
        act = self.rng.choice([0,1,2], p=p)
        # print("act", act)
        if self.parameterization == 'softmax':
            if self.optimizer == 'vanilla':
                onehot = np.zeros(3)
                onehot[act] = 1
                grad = onehot - p

            if self.optimizer == 'natural':
                # this is the minimum-norm solution to F^{-1}(gradient log pi)
                grad = -np.ones(3, dtype='float') / (3 * p[act]) + 1 / p[act] * basis_vector(act)
        elif self.parameterization == 'direct': 
            if self.optimizer == 'vanilla':
                onehot = np.zeros(3)
                onehot[act] = 1
                grad = onehot/p[act]
            else:
                raise AssertionError('invalid optimizer for direct parameterization')

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
        param_seq.append(bandit.param)
        prob_seq.append(bandit.get_prob())
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
    print("Arguments", sys.argv)

    if parameterization == 'direct':
        init_param = project(init_param)
    save_file = None  # change if you need to save the data for the runs (rewards collected)

    # Run the experiment
    param_data, prob_data = run_experiment(num_runs=num_runs,
            num_steps=num_steps, rewards=rewards, parameterization = parameterization, step_size=step_size, perturb=perturb, baseline_type=baseline_type, optimizer=optimizer, init_param=init_param, save_file=save_file)


    # Plot the learning curves: average reward vs. time
    fig = plt.figure()
    plt.grid()
    avg_rewards = []
    for i in range(num_runs):
        avg_reward_traj = np.sum(prob_data[i, :, :] * np.array(rewards), axis=1)
        avg_rewards.append(avg_reward_traj)
        if avg_reward_traj[-1] > 0.90*np.max(rewards):
            color = 'C0'
        else:
            color = 'red'
        plt.plot(avg_reward_traj, color=color, alpha=0.2)

    avg_rewards = np.array(avg_rewards)
    mean = np.mean(avg_rewards, axis=0)
    std = np.std(avg_rewards, axis=0)
    plt.plot(np.mean(avg_rewards, axis=0), color='k', linewidth=2)
    plt.ylim(0.95*np.min(rewards),1.05*np.max(rewards))
    plt.ylabel(r"$V^\pi$")
    plt.xlabel('t')
    title = f"{optimizer} {baseline_type} {perturb}"  # set to None to remove titles
    plt.title(title)

    os.makedirs('three_armed_bandit_results/', exist_ok=True)
    fig.savefig(f'three_armed_bandit_results/{optimizer}_{baseline_type}_{perturb}_eta={step_size}'.replace('.', ''), bbox_inches='tight')

    # Generate the simplex plots
    fig, tax = plot_trajectories(prob_data[:, :, :], title=title, theta_0=init_param, step_size=step_size)
    fig.savefig(f'three_armed_bandit_results/simplex_{optimizer}_{baseline_type}_{perturb}'.replace('.', ''), bbox_inches='tight')
    plt.show()
