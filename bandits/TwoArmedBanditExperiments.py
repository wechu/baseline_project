#####
# File to write down all the two-armed bandit experiments compactly
#
#
#####
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import numpy as np
from bandits.TwoArmedBandit import *
from bandits.OptimValueFunction import *
import seaborn as sns

num_runs = 300
num_steps = 300
step_size = 0.1
perturb = -1
init_param = 0
baseline_type = 'minvar'
entropy_reg = None
noise = (1.0, 1.0)
optimizer = 'natural'
parameterization = 'sigmoid'
save_file = 'results/param_data'
clip = None

param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                   step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                   optimizer=optimizer, parameterization=parameterization, entropy_reg=entropy_reg, noise=noise,
                   init_param=init_param,
                   adaptive_base=False, clip=clip,
                   save_file='results/param_data')

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
plt.title("epsilon {} noise {}".format(perturb, noise[0]))

## plot histogram of ending values
plt.figure()
plt.hist(sigmoid(param_data[:, -1]),  bins=100)
plt.ylim(0, num_runs)





## compute regret
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


## Finite horizon value function

num_steps_horizon = 40  # note if we want the values after n-steps we need to use a finite horizon td strategy
# Note that 1-step corresponds to immmediate reward, i.e. no bootstrapping, so we don't run dynamic programming for it
perturb = -1
step_size = 0.1
clip = None
optimizer = 'natural'
baseline_type = 'minvar'
entropy_reg = 0.0

values = compute_finite_horizon_value_fn(num_steps_horizon, step_size=step_size, perturb=perturb,
                                         clip=clip, optimizer=optimizer, baseline_type=baseline_type,
                                         entropy_reg=entropy_reg)

plt.plot(sigmoid(grid), values[-1, :], label="{} eps {} ent {}".format(baseline_type, perturb, entropy_reg))

plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100))
plt.legend(loc='lower right')


######
# Plain learning curves for constant baseline with divergence
num_runs = 100
num_steps = 200
step_size = 0.05  # 0.1, 0.05, 0.15
perturb = -1
init_param = -1
baseline_type = None
entropy_reg = None
optimizer = 'natural'
parameterization = 'sigmoid'
save_file = 'results/param_data/step_sizes'
clip = None

param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                   step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                   optimizer=optimizer, parameterization=parameterization, entropy_reg=entropy_reg,
                   init_param=init_param,
                   adaptive_base=False, clip=clip,
                   save_file='results/param_data')


param_data =  np.load("Neurips_results/learning_curves_constant_baselines_0.15.npy")
print("final avg performance", np.mean(sigmoid(param_data[:,-1])))
bad_threshold = 0.01
print("final proportion of bad <{}".format(bad_threshold), np.mean(sigmoid(param_data[:,-1]) < bad_threshold))

# plt.hist(np.log(np.abs(param_data[:, -1])),  bins=100)

## plot the learning curves
plt.figure()
good_lst = []
bad_lst = []
for i in range(num_runs):
    if sigmoid(param_data[i, -1]) < 0.01:
        color = 'r'
        # alpha = 0.3
        bad_lst.append(i)
    else:
        color = 'b'
        # alpha = 0.08
        good_lst.append(i)
#     plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2.5)
plt.plot(np.mean(sigmoid(param_data), axis=0), color='black', linewidth=2)


# plt.figure()
for i in good_lst:
    color = 'dodgerblue'
    alpha = 0.1
    plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2)

for i in bad_lst:
    color = 'r'
    alpha = 0.3
    plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2)

    # ax.tick_params(axis='both', which='major', labelsize=10)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

plt.ylim(-0.02, 1.02)
plt.ylabel("Prob. of choosing optimal arm", fontsize=16)
plt.xlabel("Iteration", fontsize=16)
plt.grid(True,  linestyle='-', alpha=0.1, linewidth=2)
np.save("learning_curves_constant_baselines_0.05.npy", param_data)
# plt.xlim(0,200)
# plt.title("Divergence exa")
# plt.title("epsilon {}".format(perturb, noise[0]))

# for i in range(num_runs):
#     if sigmoid(param_data[i, -1]) < 0.01:
#         color = 'r'
#         alpha = 0.3
#         bad_lst.append(i)
#     else:
#         color = 'b'
#         alpha = 0.08
#         good_lst.append(i)

# plt.figure()
# for i in range(num_runs):
#     if sigmoid(param_data[i, -1]) < 0.01:
#         color = 'r'
#         alpha = 0.2
#     else:
#         color = 'b'
#         alpha = 0.1
#     plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=1.5)
# plt.plot(np.mean(sigmoid(param_data), axis=0), color='black')
# plt.ylim(0, 1)


## plot histogram of ending values
plt.figure()
plt.hist(sigmoid(param_data[:, -1]),  bins=100)
plt.ylim(0, num_runs)


# Valentin's plotting code
param_data = np.load('Neurips_results/learning_curves_constant_baseline.npy')
num_runs=100
plt.figure(figsize=(7, 5))
good_lst = []
bad_lst = []
plt.grid(alpha=0.5)
for i in range(num_runs):
    if sigmoid(param_data[i, -1]) < 0.01:
        # color = 'r'
        # alpha = 0.3
        bad_lst.append(i)
    else:
        # color = 'b'
        # alpha = 0.08
        good_lst.append(i)
    # plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2.5)
# plt.figure()
for i in good_lst:
    color = 'dodgerblue'
    alpha = 0.15
    plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2)
plt.plot(np.mean(sigmoid(param_data), axis=0), color='black', linewidth=3)
for i in bad_lst:
    color = 'r'
    alpha = 0.25
    plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2)
    # ax.tick_params(axis='both', which='major', labelsize=10)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)
# plt.ylim(-0.02, 1.02)
plt.xlim(-1, 201)
plt.ylabel("Prob. of choosing optimal arm", fontsize=16)
plt.xlabel("Iteration", fontsize=16)
plt.savefig('bandit_committal.png', dpi=300, bbox_inches='tight')









####
# Plain learning curves for constant baseline with divergence
num_runs = 100
num_steps = 200
step_size = 0.1
perturb = -1
init_param = -1
baseline_type = None
entropy_reg = None
optimizer = 'natural'
parameterization = 'sigmoid'
save_file = 'results/param_data'
clip = None

param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                   step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                   optimizer=optimizer, parameterization=parameterization, entropy_reg=entropy_reg,
                   init_param=init_param,
                   adaptive_base=False, clip=clip,
                   save_file='results/param_data')

print("final avg performance", np.mean(sigmoid(param_data[:,-1])))
bad_threshold = 0.01
print("final proportion of bad <{}".format(bad_threshold), np.mean(sigmoid(param_data[:,-1]) < bad_threshold))

# plt.hist(np.log(np.abs(param_data[:, -1])),  bins=100)


# Plain learning curves for constant baseline with divergence (with vanilla policy gradient)
num_runs = 100
num_steps = 200
step_size = 1.5
perturb = -1
init_param = -1
baseline_type = None
entropy_reg = None
optimizer = 'regular'
parameterization = 'sigmoid'
save_file = 'results/param_data'
clip = None

param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                   step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                   optimizer=optimizer, parameterization=parameterization, entropy_reg=entropy_reg,
                   init_param=init_param,
                   adaptive_base=False, clip=clip,
                   save_file='results/param_data')

print("final avg performance", np.mean(sigmoid(param_data[:,-1])))
bad_threshold = 0.01
print("final proportion of bad <{}".format(bad_threshold), np.mean(sigmoid(param_data[:,-1]) < bad_threshold))


## plot the learning curves
plt.figure()
good_lst = []
bad_lst = []
for i in range(num_runs):
    if sigmoid(param_data[i, -1]) < 0.01:
        color = 'r'
        # alpha = 0.3
        bad_lst.append(i)
    else:
        color = 'b'
        # alpha = 0.08
        good_lst.append(i)
    # plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2.5)

plt.figure()
plt.plot(np.mean(sigmoid(param_data), axis=0), color='black', linewidth=2)
for i in good_lst:
    color = 'dodgerblue'
    alpha = 0.1
    plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2)

for i in bad_lst:
    color = 'r'
    alpha = 0.3
    plt.plot(sigmoid(param_data[i, :]), color=color, alpha=alpha, linewidth=2)

    # ax.tick_params(axis='both', which='major', labelsize=10)
plt.yticks(fontsize=16)
plt.xticks(fontsize=16)

plt.ylim(-0.02, 1.02)
plt.ylabel("Prob. of choosing optimal arm", fontsize=16)
plt.xlabel("Iteration", fontsize=16)







######
# Adding noise to the rewards
num_runs = 300
num_steps = 300
step_size = 0.1
init_param = 0
baseline_type = 'minvar'
optimizer = 'natural'
parameterization = 'sigmoid'
save_file = 'results/param_data/exp_noise/'
clip = None

for noise_std in [0.5]:
    noise = (noise_std, noise_std)
    for perturb in (-1.1, 1.1):
        param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                           step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                           optimizer=optimizer, parameterization=parameterization, noise=noise,
                           init_param=init_param, clip=clip, adaptive_base=False,
                           save_file=save_file, save_vars=("noise", "perturb"))

        ## plot the learning curves
        plt.figure()
        for i in range(num_runs):
            plt.plot(sigmoid(param_data[i, :]), color='b', alpha=0.08)
        plt.plot(np.mean(sigmoid(param_data), axis=0), color='black')
        plt.ylim(0, 1)
        plt.title("epsilon {} noise {}".format(perturb, noise[0]))
        plt.ylabel('Prob. of right')
        plt.xlabel("Steps")


######
# Adding entropy
num_runs = 300
num_steps = 300
step_size = 0.1
init_param = -3
clip = None
baseline_type = 'minvar'
optimizer = 'natural'
parameterization = 'sigmoid'
save_file = 'results/param_data/exp_entropy/'

for entropy_reg in (0.1, 0.2, 0.5):
    for perturb in [-1.1]:
        param_data = run_experiment(num_runs=num_runs, num_steps=num_steps,
                           step_size=step_size, perturb=perturb, baseline_type=baseline_type,
                           optimizer=optimizer, parameterization=parameterization, entropy_reg=entropy_reg,
                           init_param=init_param, clip=clip,
                           save_file=save_file, save_vars=["entropy_reg", "perturb"])
        ## plot the learning curves
        plt.figure()
        for i in range(num_runs):
            plt.plot(sigmoid(param_data[i, :]), color='b', alpha=0.08)
        plt.plot(np.mean(sigmoid(param_data), axis=0), color='black')
        plt.ylim(0, 1)
        plt.title("epsilon {} entropy {}".format(perturb, entropy_reg))
        plt.ylabel('Prob. of right')
        plt.xlabel("Steps")

        # plt.savefig()

num_steps_horizon = 40
# perturb = -1
step_size = 0.1
clip = None
optimizer = 'natural'
baseline_type = 'minvar'
entropy_reg = 0.0

for perturb in [-1, 0, 1]:
    plt.figure()
    for entropy_reg in [0.0, 0.1, 0.2, 0.5, 1.0]:
        values = compute_finite_horizon_value_fn(num_steps_horizon, step_size=step_size, perturb=perturb,
                                                 clip=clip, optimizer=optimizer, baseline_type=baseline_type,
                                                 entropy_reg=entropy_reg)

        plt.plot(sigmoid(grid), values[-1, :], label="eps {} entropy {}".format(perturb, entropy_reg))

    plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='black')
    plt.legend(loc='lower right')



#####
# Effect of clipping

num_steps_horizon = 40
perturb = 0
step_size = 0.1
optimizer = 'natural'
baseline_type = 'minvar'
entropy_reg = 0.0

plt.figure()
for clip in [None, 0.1, 0.3, 1, 3, 10]:
    values = compute_finite_horizon_value_fn(num_steps_horizon, step_size=step_size, perturb=perturb,
                                             clip=clip, optimizer=optimizer, baseline_type=baseline_type,
                                             entropy_reg=entropy_reg)

    plt.plot(sigmoid(grid), values[-1, :], label="eps {} clip {}".format(perturb, clip))

plt.plot(np.linspace(0, 1, 100), np.linspace(0, 1, 100), color='black')
plt.legend(loc='lower right')





#####
# Using the value function as a baseline




#####
# Regular SGD



#####
#






