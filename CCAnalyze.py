import numpy as np
import pickle
import collections
from collections import OrderedDict as OD
import itertools
import matplotlib.pyplot as plt
#


path = 'res/11_05_13_2020/1604601182_/'

with open(path + "config.pkl", 'rb') as f:
    config = pickle.load(f)

print(config.id)

alg = 'reinforce'
sweep_params_dict = collections.OrderedDict(
    list(config.shared_sweep_params.items()) + list(config.algs_sweep_params[alg].items()))

hyperparam_names = list(sweep_params_dict.keys())
hyperparam_tuples = list(itertools.product(*list(sweep_params_dict.values())))
num_hyperparam = len(hyperparam_tuples)
# list of all hyperparameter settings, note that it is still in the same order as generated in CreateJobs.py

#### Merge runs when parallelizing runs
for i_hyp in range(num_hyperparam):
    for logged_value in config.logged_values:
        result_lst = []

        if logged_value == 'returns':
            logged_value = 'all_returns'

        for i_run in range(config.num_runs):
            result = np.load(path + 'Runs/{}_{}/run_{}/{}.npy'.format(alg, i_hyp, i_run, logged_value))
            result_lst.append(result)

        aggregate_result = np.concatenate(result_lst, axis=0)
        print(i_hyp, aggregate_result.shape)
        np.save(path + "Runs/{}_{}/{}.npy".format(alg, i_hyp, logged_value), aggregate_result)



###

# Load results
# dictionary with the format { hyperparameter tuple: [logged_value1, logged_value2, ...] }
all_results = OD()
for i_hyp in range(num_hyperparam):
    result_lst = []
    # result_lst.append(hyperparam_tuples[i_hyp])
    for logged_value in config.logged_values:
        if logged_value == 'returns':
            result = np.load(path + 'Runs/{}_{}/all_returns.npy'.format(alg, i_hyp))
            result_lst.append(result)
        if logged_value == 'action_entropy_trajectory':
            result = np.load(path + 'Runs/{}_{}/action_entropy_trajectory.npy'.format(alg, i_hyp))
            result_lst.append(result)

        if logged_value == 'state_visitation_entropy_online':
            result = np.load(path + 'Runs/{}_{}/state_visitation_entropy_online.npy'.format(alg, i_hyp))
            result_lst.append(result)

        if logged_value == 'state_visitation_entropy_eval':
            result = np.load(path + 'Runs/{}_{}/state_visitation_entropy_eval.npy'.format(alg, i_hyp))
            result_lst.append(result)

    all_results[hyperparam_tuples[i_hyp]] = result_lst
    # all_results.append(result_lst)


# all_results is a dict that contains a tuple of hyperparameters as the key and the value is the list of logged results
# (each element of the list is one logged value e.g. returns)
print(hyperparam_names)
print(len(all_results))
print(all_results.keys())
print(list(all_results.values())[0][2])




# data_ent should be  N_seeds x N_baselines x T array
step_size = 0.1
baselines=[-1, -0.5, 0, 0.5, 1]

metric_index = 0
for metric_index in range(4):
    results = []
    for i in range(len(hyperparam_tuples)):
        hyp = hyperparam_tuples[i]
        if hyp[3] == step_size:
            print(hyp)
            results.append(all_results[hyp][metric_index])  # choose index based on 'returns', 'action_entropy_trajectory', etc.

    results = np.array(results)
    results = results.transpose([1, 0, 2])
    data_ent = results
    N = 100

    # Valentin's plots
    from matplotlib.pyplot import Subplot
    fig = plt.figure(figsize = (9, 6.5))
    ax = Subplot(fig, 111)
    fig.add_subplot(ax)
    # ax.axis["right"].set_visible(False)
    # ax.axis["top"].set_visible(False)
    save_freq = 200
    vec = np.arange(data_ent.mean(0).T[:, 0].shape[0]) * save_freq

    # vec =
    # vec = np.logspace(0, np.log10(50000), data_ent.mean(0).T[:, 0].shape[0])
    ylabels = ['return', 'Action entropy trajectory', 'Online state visitation entropy', 'Offline state visitation entropy']
    for i in range(5):
        # color = (i/5, 0.2, 1-i/5)
        mean = data_ent[:, i, :].mean(0)
        std = data_ent[:, i, :].std(0)/np.sqrt(N)
        plt.plot(vec, mean, label = str(baselines[i]), linewidth=4)
        plt.fill_between(vec, mean-std, mean+std,  alpha=0.10)
        # plt.ylabel(r'$H(\pi)$')
        plt.ylabel(ylabels[metric_index])
        plt.xlabel('t')
        # plt.xscale('log')
    plt.legend()

    # plt.savefig('entropy_pi.pdf', dpi=300, bbox_inches='tight')




def plot_cumulative_best_goal_reaches_window(data, window=1000):
    plt.figure()
    data_reach_goal = np.cumsum(data > 0.99, axis=1)

    for i in range(len(data_reach_goal)):
        one_run = (data_reach_goal[i, window:] - data_reach_goal[i, :-window]) / float(window)
        plt.plot(one_run, color='blue', alpha=0.1)

def running_mean(x, N):
    cumsum = np.cumsum(x > 0.99)
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def running_mean(x, N):

    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def plot_smoothed_learning_curves(data, smooth_window=5):
    ''' plots all learning curves in data
    assumes each row in data is one learning curve '''
    plt.figure()
    for i in range(len(data)):
        plt.plot(running_mean(data[i], smooth_window), color='blue', alpha=0.1)


def plot_cumulative_best_goal_reaches(data):
    plt.figure()
    data_reach_goal = np.cumsum(data > 0.99, axis=1)

    for i in range(len(data_reach_goal)):
        plt.plot(data_reach_goal[i], color='blue', alpha=0.1)


query_hyperparam = ['SGD', 0.99, 100, 10**(-1), 1e-2, 0.0]

name_sweep = 'perturb'
idx_sweep = hyperparam_names.index(name_sweep)


#
i = 0
for h_setting in sweep_params_dict[name_sweep]:
    i += 1
    if i % 1 == 0:
        sweep_query = query_hyperparam.copy()
        sweep_query[idx_sweep] = h_setting

        # plot_smoothed_learning_curves(all_results[tuple(sweep_query)][0], smooth_window=50)  # index 0 denotes 'returns' here
        plot_cumulative_best_goal_reaches(all_results[tuple(sweep_query)][0])  # index 0 denotes 'returns' here

        plt.title("{} {}".format(name_sweep, round(h_setting, 3))) # to avoid weird floats, use round()
        plt.ylim(0, 2000)
        # plt.ylim(0.7, 1)
        # 1d histogram at the end
        # plt.figure()
        # plt.hist(np.mean(all_results[tuple(sweep_query)][0][:, -50:],axis=0) , bins=100, range=[0.8, 1], )
        # plt.ylim(0, 100)
        # plt.title("{} {}".format(name_sweep, h_setting))

        # 2d histogram
        plt.figure()
        num_runs = 100
        window =50
        running_means =  np.array([running_mean(all_results[tuple(sweep_query)][0][i], window) for i in range(num_runs)])

        plt.hist2d(np.array([np.arange(0, running_means.shape[1]) for i in range(num_runs)]).flatten(), running_means.flatten(), bins=100,
                   vmin=0, vmax=1000)
        plt.ylim(0.7, 1.0)
        plt.colorbar()
        plt.title("{} {:.3f}".format(name_sweep, h_setting))






from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')






