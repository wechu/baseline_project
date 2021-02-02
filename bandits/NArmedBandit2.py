###
#
#
#
#
#
# with parameterization where each action has a parameter p
###

import numpy as np
from numba import jit
import matplotlib.pyplot as plt

np.set_printoptions(precision=3, linewidth=500)

num_dimensions = 100


def generate_bandit_problem(setting='pos'):
    ''' 2-armed bandit problem
    Returns f, grad_f, sgd_f, var_sgd functions
    f(x): returns the function value
    grad_f(x): returns the true gradient
    sgrad(x): returns a sample of the gradient
    m2_sgrad(x): returns the second moment of the norm of the stochastic gradient
    Note: x represents the prob. of choosing arm 1'''

    if setting == 'pos':
        epsilon = 1
    elif setting == 'neg':
        epsilon = -1
    elif setting == 'opt':
        epsilon = 0

    base_r = np.zeros(num_dimensions)
    base_r[0] = 1

    # @jit(nopython=True)
    def f(p):  # function to maximize: 2-armed bandit
        return np.sum(base_r * p)

    # @jit(nopython=True)
    def grad_f(p):
        return base_r

    def optimal_baseline(p):
        epsilon = 1e-12
        p = np.clip(p, epsilon, 1-epsilon)
        numerator = np.sum(base_r / p)
        denominator = np.sum(1 / p)
        return numerator / denominator

    # @jit(nopython=True)
    def sgrad(p):
        act = np.random.choice(np.arange(0, num_dimensions), p=p)
        one_hot = np.zeros(len(p))
        one_hot[act] = 1
        return (base_r[act] - optimal_baseline(theta) + epsilon) / p[act] * one_hot
        # return (base_r[act] + epsilon) / p[act] * one_hot

        #
        # almost_zero = 1e-12
        # if p[act] > almost_zero:
        #     return (r[act]-optimal_baseline(theta)) / p[act]
        # else:
        #     return (r[act]- / almost_zero

    # @jit(nopython=True)
    def m2_sgrad(p):
        # returns the norm of the second moment of the stochastic gradient
        return np.sum( np.square(base_r) / p)

    return f, grad_f, sgrad, m2_sgrad

# @jit(nopython=True)
# def project(x):
#     # taken from paper by Wang et al.
#     u = sorted(x, reverse=True)
#     rho = np.argmax([ np.maximum(u[j] + 1/j * (1 - np.sum(u[:j+1])), 0) for j in range(len(u))])
#     lmbda = 1 / rho * (1 - np.sum(u[:rho+1]))
#     return np.clip(np.array(x) + lmbda, 0, 1)

#@jit(nopython=True)
def project_to_probsimplex(x):
    # taken from Chen and Ye (2011)
    # l2 projection to probability simplex
    u = np.array(sorted(x))
    for i in range(len(x)-2, -1, -1):
        t = (np.sum(u[i+1:]) - 1) / (len(x) - 1 - i)
        if t >= u[i]:
            t_hat = t
            break
    else:
        t_hat = (np.sum(u) - 1)/ len(x)

    return np.clip(x - t_hat, 0, 99)  # 99 doesn't matter, it's max(0, x - t_hat)

# f, grad_f, sgrad_fn, m2_sgrad_fn = generate_bandit_problem('pos')
#
# num_steps = 100
# num_repeats = 1000
# init_step_size = 0.01
#
# init_theta = 1/num_dimensions* np.ones(num_dimensions-1)
#
# all_theta_results = []
# all_theta_results_path = []
#
# print(init_theta)
# theta_results = []
# theta_results_path_reps = []
#
# for rep in range(num_repeats):
#     theta_results_path = []
#     theta = init_theta
#
#     for i in range(num_steps):
#         theta_results_path.append(theta)
#         # do a projected sgd step
#         theta = project(theta + init_step_size * sgrad_fn(theta))
#
#         # print(i, theta)E
#     theta_results.append(theta)  # record the final parameter value
#     theta_results_path_reps.append(theta_results_path)
#
# all_theta_results.append(theta_results)  # appends for each init_theta value
# all_theta_results_path.append(theta_results_path_reps)


for setting in ('opt', 'neg', 'pos'):
    f, grad_f, sgrad_fn, m2_sgrad_fn = generate_bandit_problem(setting)

    init_thetas = [1/num_dimensions * np.ones(num_dimensions)]
    num_steps = 1000
    num_repeats = 10
    init_step_size = 0.005

    # all_theta_results = []
    all_theta_results_path = []
    for init_theta in init_thetas:
        print(init_theta)
        # theta_results = []
        theta_results_path_reps = []

        for rep in range(num_repeats):
            theta_results_path = []
            theta = init_theta

            for i in range(num_steps):
                theta_results_path.append(theta)
                # do a projected sgd step

                sgd = sgrad_fn(theta)
                # print('grad', sgd)
                theta = theta + init_step_size * sgd
                # print('raw', theta)
                theta = project(theta)
                # p = np.append(theta, 1 - np.sum(theta))
                # projected_p = project(p)
                # print(p)
                # print("proj", projected_p)
                # theta = projected_p[:len(theta)]

                # theta = project()
                print(i, theta)
                # print(i, theta)E
            # theta_results.append(theta)  # record the final parameter value
            theta_results_path = np.stack(theta_results_path)
            theta_results_path_reps.append(theta_results_path)

        # all_theta_results.append(theta_results)  # appends for each init_theta value
        all_theta_results_path.append(theta_results_path_reps)

    index_query = 0
    # plt.figure()
    # plt.hist(all_theta_results_path[index_query][-1][0], bins=20, range=[0,1])  # the last [0] is for the prob of choosing arm 0 (the best one)
    # plt.ylim(0, num_repeats)
    # plt.title(setting)

    # plt.figure()
    # num_lines = 10
    # for i in np.random.randint(0, num_repeats, num_lines):
    #     plt.plot(all_theta_results_path[index_query][i][:, 0], color='b', alpha=0.2)
    # plt.ylim(0, 1)
    # plt.title(setting)
    #
    # # plt.xlim(0, 20)
    # # plt.show()
    # all_theta_results_path = np.array(all_theta_results_path)
    # print("num zeros:", np.sum(np.isclose(all_theta_results_path[index_query, :, -1,0],0) ))
    # plt.show()
    np.save('results/Narmed_paths_{}'.format(setting), all_theta_results_path)

    # variances = np.var(all_theta_results_path[index_query], axis=0)
    # np.save("results/vars_{}".format(setting), variances)


### Plotting
import matplotlib.pyplot as plt
import numpy as np

num_repeats = 10
f1 = plt.figure(1)
f2 = plt.figure(2)
colors = {'neg': 'b', 'pos': 'g', 'split':'r', 'pos_zero':'gold', 'opt': 'orange'}
for setting in ("neg", "pos", 'opt'):
    plt.figure(1)
    all_results = np.load("results/Narmed_paths_{}.npy".format(setting))

    num_lines = 10
    index_query = 0
    for i in range(10): #np.random.randint(0, num_repeats, num_lines):
        plt.plot(all_results[index_query][i][:, 0], color=colors[setting], alpha=0.3)  # 0 is the variable
    plt.ylim(0, 1)
    print(setting, "num zeros:", np.sum(np.isclose(all_results[index_query, :, -1,0],0) ))
    print(all_results[index_query, :, -1,0])
plt.show()
    # variances = np.load('results/vars_{}.npy'.format(setting))
#     plt.figure(1)
#     plt.plot(variances, color=colors[setting], linewidth=2)
#
#     plt.figure(2)
#     diff_var = variances[1:] - variances[:-1]
#     def smooth(x, N):
#         return np.convolve(x, np.ones((N,)) / N, mode='valid')
#     diff_var = smooth(diff_var, 100)
#     plt.plot(diff_var, color=colors[setting])
#
# plt.figure(1)
# plt.title('vars')
# # plt.ylim(0, 0.0013)
#
# plt.figure(2)
# plt.title('diff_var')
# plt.show()
#