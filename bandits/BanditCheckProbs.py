###
# Running simple simulations to check 'catastrophe' probabilities
#
# +eps is good baseline, -eps is bad baseline
#
###
import matplotlib.pyplot as plt
import numpy as np
from utils import *

## Various decay rates for the prob of taking the good action
def rec_sigmoid(x, alpha, epsilon=1):
    # epsilon is how much the baseline underestimates the optimal one
    clip_tolerance = 1e-10
    # x = np.clip(x, 1e-9, 1-1e-9)
    return sigmoid(sigmoid_inv(x) + alpha * (1 - epsilon / (1-np.clip(x, clip_tolerance, 1-clip_tolerance))))

def rec_logit_sigmoid_constant_baseline(x, alpha, baseline):
    clip_tolerance = 1e-10
    return sigmoid(sigmoid_inv(x) + alpha * baseline / (1 - np.clip(x, clip_tolerance, 1-clip_tolerance)))

def rec_logistic(x, alpha):
    return x*(1-alpha*x)

def rec_exponential(x, alpha):
    return x * math.exp(-alpha)




## checking decay rate
x_lst = []

x = 0.5
x_lst.append(x)
for i in range(100):
    x = rec_logistic(x, 0.3)
    x_lst.append(x)
    print(x)

x_lst = np.log(1 - np.array(x_lst))
# y_lst = np.log(1 - x_lst
plt.plot(x_lst)
# plt.show()

### check catastrophe prob for each x_0
x_check = np.arange(0, 1, 0.01)  # initial prob of taking the optimal action
m_check = [1, 2, 3, 4, 5, 10, 100, 1000]
alpha = 0.3
epsilon = -1

plt.figure()
plt.title("alpha {} eps {}".format(alpha, epsilon))
for m in m_check:
    results = []
    for x_0 in x_check:

        prod = 1 - x_0
        x = x_0
        for i in range(m-1):
            x = rec_sigmoid(x, alpha, epsilon)
            prod *= (1-x)

        results.append(prod)

    plt.plot(results)


def get_cat_prob(x_0, alpha, epsilon, T):
    prod = 1 - x_0
    x = x_0
    for i in range(T-1):
        # x = rec_logit_sigmoid_constant_baseline(x, alpha, epsilon)
        x = rec_logit_sigmoid_constant_baseline(x, alpha, epsilon)
        prod *= (1-x)
    return prod

def cat_prob_bound1(alpha, baseline):
    # this is the one Nicolas derived on prob for infinite left
    # assume we start at x=0.5
    return 0.5 * np.exp( np.log(1- np.exp(alpha*baseline) ) / (1-np.exp(alpha*baseline)))

def cat_prob_bound12(alpha, baseline):
    return 0.5 * (1 - np.exp(alpha*baseline) )**(-1/(alpha*baseline))

def cat_prob_bound2(p_0, alpha, T):
    # assumes baseline is the optimal one perturbed by -1
    return ( (1-p_0) / (1 - p_0 + alpha * p_0 * T))**(1 / alpha)
## comparing empirical probabilities to theoretical ones
# p_0 = 0.3
alpha = 0.2
baseline = -2
print(get_cat_prob(0.5, alpha, baseline, 1000),  cat_prob_bound1(alpha, baseline), cat_prob_bound12(alpha, baseline))
# print(get_cat_prob(p_0, alpha, baseline, 100),  cat_prob_bound2(p_0, alpha, 100))

# ### Try to define reasonable one step values
# x_check = np.arange(0, 1, 0.01)
# def value(x, alpha, epsilon, cat_probs, x_check):
#     # return of x + return improvement in 1 step
#     return (x + alpha - alpha**2 * (1 + epsilon**2 / (x*(1-x)))) * (1 - cat_probs[list(x_check).index(x)])
#
# def value2(x, alpha, epsilon, cat_prob):
#     # return of x + return improvement in 1 step
#     return (x + alpha - alpha**2 * (1 + epsilon**2 / (x*(1-x)))) * (1 - cat_prob)
#
#
# values = []
# i=0
# for x in x_check:
#     values.append(value(x, 0.1, 1, results, i))
#     i+=1
#
# plt.figure()
# plt.plot(values)
#
# ### Check epsilon curve
# x = 0.5
# eps_check = np.arange(-6, 6, 0.1)
# values = []
#
# for eps in eps_check:
#     values.append(value2(x, 0.3, eps, get_cat_prob(x, 0.3, eps, 100)))
#
# plt.figure()
# plt.plot(values)
#


### Check sequences of good and bad actions
# check sequences of 16 actions
settings = [[1, 2]*8,
            [1, 1, 2, 2]*4,
            [1, 1, 1, 1, 2, 2, 2, 2]*2,
            [1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2],
            [2, 1]*8,
            [2, 2, 1, 1]*4,
            [2, 2, 2, 2, 1, 1, 1, 1]*2,
            [2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]]


from bandits.TwoArmedBandit import *
init_param = 0
perturb = 1
step_size = 0.3

results = []  # (log_probs_lst, performance)
for act_seq in settings:
    env = Bandit(1, 0, init_param, perturb_baseline=perturb)

    log_probs_lst= []
    for act in act_seq:
        prob = env.do_sgd_step_action(step_size, act)
        log_probs_lst.append(np.log(prob))
    results.append([log_probs_lst, env.get_prob()])

print("epsilon", perturb, [round(x[1], 3) for x in results])
final_perfs = np.array([x[1] for x in results])
probs = np.array([np.exp(np.sum(x[0])) for x in results])



### Get full distribution of parameter values over k steps
from bandits.TwoArmedBandit import *
from utils import *
num_steps = 16
step_size = 0.2
perturb = -1.1
init_param = sigmoid_inv(0.5)
optimizer = 'natural'
parameterization = 'sigmoid'

def get_distribution(num_steps, step_size, perturb, init_param, optimizer, parameterization, include_actions=False):
    ''' Returns the distribution over parameters after num_steps steps as a list of tuples (param, prob)
    include_actions: If true, also returns the sequence of actions leading to the parameter value in the tuple (param, prob, actions) '''
    env = Bandit(1, 0, init_param, perturb_baseline=perturb, optimizer=optimizer, parameterization=parameterization)
    # Note we can change both rewards to 0 to obtain a martingale

    distribution = [(init_param, 0.0)]
    if include_actions:
        distribution = [(init_param, 0.0, '')]

    for i in range(num_steps):
        new_distribution = []

        for entry in distribution:
            x = entry[0]
            log_prob = entry[1]
            updates = env.get_possible_gradients(x, return_next_params=True, step_size=step_size)

            # computes the next parameters and probability for each possible update
            if include_actions:
                action_seq = entry[2]
                act = 1
                for new_x, new_prob in updates:
                    new_distribution.append((new_x, log_prob + np.log(new_prob), action_seq + str(act)))
                    act += 1
            else:
                for new_x, new_prob in updates:
                    new_distribution.append((new_x, log_prob + np.log(new_prob)))

        distribution = new_distribution

    # distribution.sort(key= lambda x: x[0])

    # temp = np.exp([x[1] for x in distribution])
    #
    #
    # distribution = np.array(distribution).astype('float64')
    # distribution[:, 2]  = distribution[:, 2].astype('int32')
    if include_actions:
        distribution = [(a, np.exp(b), c) for a, b, c in distribution]
    else:
        distribution = np.array(distribution)
        distribution[:, 1] = np.exp(distribution[:,1])  # uses log probs to avoid numerical issues
    # distribution.astype('str')
    return distribution


## Check distribution
# here we compare the outcomes of all sequences of increases/decreases for pos/neg baselines
#
dist = get_distribution(num_steps, step_size, perturb, init_param, optimizer, parameterization, include_actions=True)
from collections import OrderedDict
dist = OrderedDict((c, (a,b)) for a, b, c in dist) # contains +eps

dist2 = get_distribution(num_steps, step_size, -perturb, init_param, optimizer, parameterization, include_actions=True)
from collections import OrderedDict
dist2 = OrderedDict((c, (a,b)) for a, b, c in dist2) # contains -eps

# we need to invert the action sequences because action 1 either increases or decreases theta depending on whether the baseline was pos or neg
# so, in the comparisons, to match the number of increases, we need to swap the actions
def invert(action_seq):
    # switches 1 and 2
    s = ""
    for a in action_seq:
        if a == '1':
            s += '2'
        else:
            s += '1'
    return s
lst = []
for k in dist.keys():  # compare end parameter value for each sequence
    lst.append(dist[k][0] > dist2[invert(k)][0])
    if dist[k][0] < dist2[invert(k)][0]:
        print(k, dist[k][0], dist2[invert(k)][0])

print(np.mean(lst), np.sum(lst))

# just check the most probable action sequences
dist.sort(key=lambda x: x[1], reverse=True)
dist2.sort(key=lambda x: x[1], reverse=True)

## Check distribution after k steps
# this gets the full distribution

# gets the average return after k steps for a grid of values of x
results = []  # this is the mean return
xs = np.arange(0.01, 1, 0.01)
for x in xs:
    distribution = get_distribution(num_steps, step_size, perturb, sigmoid_inv(x), optimizer, parameterization, False)
    results.append(np.sum(sigmoid(distribution[:,0])*distribution[:,1]))
plt.plot(xs, results)
print(results)

# plots the distribution after k steps starting at some initial parameter value
distribution = get_distribution(num_steps, step_size, perturb, 0, optimizer, parameterization, False)
plt.figure()
plt.xlim((0,1))
plt.ylim((0,0.5))
plt.title("num steps {} epsilon {}".format(num_steps, perturb))
plt.hist(sigmoid(distribution[:,0]), weights=distribution[:,1],bins=20)  # plot the resulting probs
plt.ylabel('Probability')
plt.xlabel('Prob. of right action')
# plt.hist(np.log(np.abs(distribution[:,0])), weights=distribution[:,1], bins=50)  # plot the raw parameter values
print("mean", np.sum(sigmoid(distribution[:,0])*distribution[:,1]))




#### baseline