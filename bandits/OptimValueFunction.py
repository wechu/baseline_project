######
# Computing the value function for the optimization problem where states are parameters
# This is quite similar to computing the distribution over parameters
#
######

from bandits.TwoArmedBandit import *
import numpy as np
### Use dynamic programming for k steps
# we discretize in parameter space

grid_limit = 10
grid_step = 0.01
grid = np.arange(-grid_limit, grid_limit+grid_step/2, grid_step)

# @jit
def project_to_grid(x, grid_limit, grid_step):
    # returns the closest x-value on the grid
    return np.clip(round(x / grid_step) * grid_step, -grid_limit, grid_limit)
# @jit
def convert_to_index(x, grid, grid_step):
#     returns the index in the grid of x. (Assumes that x is in the grid)
    return np.where(np.abs(grid - x) < grid_step/10)[0][0]


### finite horizon value function

def compute_finite_horizon_value_fn(num_steps_horizon, step_size, perturb, clip, optimizer, baseline_type, entropy_reg):
    bandit = Bandit(1, 0, init_param=None, perturb_baseline=perturb, clip=clip, optimizer=optimizer,
                 baseline_type=baseline_type, entropy_reg=entropy_reg)

    values = np.zeros((num_steps_horizon, grid.shape[0]), dtype='float64')
    # values[:, -1] = 1  # the last state is given value 1 and the first state is given value 0
    values[0, :] = sigmoid(grid)  # initialize values of the first iteration to be equal to the probability of picking action 1

    for horizon in range(1, num_steps_horizon):  # note we skip the first one, since it is just the immediate reward
        temp_values = values[horizon, :].copy()

        for i in range(grid.shape[0]): # one pass of dynamic programming through the states
            # if i == 0:  # we assume the values of the first and last state are 0 and 1 respectively
            #     values[horizon, i] = 0
            # elif i == (grid.shape[0]-1):
            #     values[horizon, i] = 1
            # else:
            # for other states, we do a dynamic programming update
            updates = bandit.get_possible_gradients(grid[i], return_next_params=True, step_size=step_size)
            new_value = 0.0
            for (next_x, prob) in updates:
                proj_x = project_to_grid(next_x, grid_limit, grid_step)
                # print(proj_x)
                next_val = values[horizon-1, convert_to_index(proj_x, grid, grid_step)]

                # print(proj_x)
                new_value += prob * next_val
            values[horizon, i] = new_value
        # print(values != temp_values)
        change = np.max(np.abs(temp_values - values[horizon, :]))
        print('horizon {} change in values {}'.format(horizon+1, change))
    return values


#### infinite horizon
error_tolerance = 1e-2
def compute_infinite_horizon_value_fn(discount, step_size, perturb, clip, optimizer, baseline_type, entropy_reg):
    env = Bandit(1, 0, init_param=None, perturb_baseline=perturb, clip=clip, optimizer=optimizer,
                 baseline_type=baseline_type, entropy_reg=entropy_reg)

    values = np.zeros(grid.shape[0], dtype='float64')

    rewards = 'probs'
    if rewards == 'sparse':
        values[-1] = 1


    change = 999

    while change > error_tolerance:
        temp_values = values.copy()

        for i in range(grid.shape[0]): # one pass of dynamic programming through the states
            # TODO we remove the constraints at 0 and 1 if we consider the cumulative reward over time (i.e. something like regret)
            # if i == 0:  # we assume the values of the first and last state are 0 and 1 respectively
            #     # values[i] = 0
            #     values[i] = discount*values[i] + 0
            # elif i == (grid.shape[0]-1):
            #     # values[i] = 1
            #     values[i] = discount*values[i] + 1
            # else:
            # for other states, we do a dynamic programming update
            if rewards == 'sparse':
                if i == 0:  # we assume the values of the first and last state are 0 and 1 respectively
                    values[i] = 0
                    continue
                    # values[i] = discount*values[i] + 0
                elif i == (grid.shape[0]-1):
                    values[i] = 1
                    continue

            updates = env.get_possible_gradients(grid[i], return_next_params=True, step_size=step_size)
            new_value = 0.0

            for j in range(len(updates)):

                next_x, prob = updates[j]
                proj_x = project_to_grid(next_x, grid_limit, grid_step)
                # print(proj_x)
                next_val = values[convert_to_index(proj_x, grid, grid_step)]

                if rewards == 'sparse':
                    new_value += prob * next_val
                elif rewards == 'probs':
                    reward = 1 if j == 0 else 0  # assumes the first action gives 1, the rest 0
                    # sigmoid(next_x) as the reward is interesting too (but probably wrong)
                    new_value += prob * (discount * next_val + reward)

            values[i] = new_value
        # print(values != temp_values)
        change = np.max(np.abs(temp_values - values[:]))
        print('change in values {}'.format(change))
    return values




if __name__ == "__main__":
    #####
    # finite horizon

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

    plt.plot(sigmoid(grid), values[-1,:], label="{} eps {} ent {}".format(baseline_type, perturb, entropy_reg))

    plt.plot(np.linspace(0, 1, 100), np.linspace(0,1, 100))
    plt.legend(loc='lower right')


    #####
    # infinite horizon
    perturb = 1
    step_size = 0.1
    discount = 0.95
    clip = None
    optimizer = 'natural'
    baseline_type = 'minvar'
    entropy_reg = None

    values = compute_infinite_horizon_value_fn(discount, step_size=step_size, perturb=perturb,
                                               clip=clip, optimizer=optimizer, baseline_type=baseline_type,
                                               entropy_reg=entropy_reg)

    plt.plot(sigmoid(grid), values*(1-discount), label="{} eps {}".format(baseline_type, perturb))
    plt.plot(np.linspace(0, 1, 100), np.linspace(0,1, 100))

    plt.legend(loc='lower right')

    # np.save('values.npy', values)