import numpy as np
import matplotlib.pyplot as plt
from utils import *

step_size = 0.003
perturbs = np.arange(-6, 6.1, 0.5)
results = {}

for perturb in perturbs:
    results[(step_size, perturb)] = sigmoid(np.load("results/param_data_step{}_pert{}_nat_sigmoid.npy".format(step_size, perturb)))


summary_stats = {'final_mean': {}, 'final_std': {}, 'final_quart1': {}, 'final_quart3': {}}


for k, data in results.items():
    summary_stats['final_mean'][k] = np.mean(data[:, -1])
    summary_stats['final_std'][k] = np.std(data[:, -1])
    # summary_stats['final_quart1'][k] = np.quantile(data[:, -1], 0.25)
    # summary_stats['final_quart3'][k] = np.quantile(data[:, -1], 0.75)

means = np.array([summary_stats['final_mean'][(step_size, perturb)] for perturb in perturbs])
stds = np.array([summary_stats['final_std'][(step_size, perturb)] for perturb in perturbs])

# quart1 = np.array([summary_stats['final_quart1'][(step_size, perturb)] for perturb in perturbs])
# quart3 = np.array([summary_stats['final_quart3'][(step_size, perturb)] for perturb in perturbs])

plt.figure()
plt.plot(perturbs, means, color='b')
plt.fill_between(perturbs, means+stds, means-stds, color='b', alpha=0.25)
plt.grid()
plt.title(step_size)
# plt.axvline(x=0.0, color='black', alpha=0.5)
# plt.axhline(y=1.0, color='black', alpha=0.5)
# plt.plot(perturbs, means+stds, color='g')
# plt.plot(perturbs, means-stds, color='g')
# plt.plot(perturbs, quart1, color='g')
# plt.plot(perturbs, quart3, color='g')
plt.show()


# plt.scatter()
#
# plt.scatter(x, y, facecolors='none', edgecolors='b')




########### Variance in two steps analytically

