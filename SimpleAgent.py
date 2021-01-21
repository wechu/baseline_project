###
# Gridworld environment
#
#
###
import SimpleMDP
from utils import *
import time


class PGAgent:
    def __init__(self, num_actions, discount, baseline_type=None, seed=None, env=None):
        # tabular
        if env.name.lower() in ['gridworld', 'fourrooms']:
            self.param = np.zeros(shape=[env.gridsize[0], env.gridsize[1], num_actions])  # height x width x num_actions
            self.avg_returns = np.zeros(shape=[env.gridsize[0], env.gridsize[1]])  # used to save avg return from each state (as a baseline)
            self.visit_counts = np.zeros(shape=[env.gridsize[0], env.gridsize[1]])  # used to count visits to each state
        elif env.name.lower() == 'binomialtree':
            # note that the some entries of these arrays are unused, since we only need the lower triangular part to represent the tree
            self.param = np.zeros(shape=[env.depth, env.depth, num_actions])
            self.avg_returns = np.zeros(shape=[env.depth, env.depth])
            self.visit_counts = np.zeros(shape=[env.depth, env.depth])  # used to count visits to each state

        self.num_actions = num_actions
        self.baseline_type = baseline_type  # "avg", "minvar", None

        self.discount = discount
        self.rng = np.random.RandomState(seed)
        self.env = env  # this is needed to compute the minimum-variance baseline

        self.q_values = None  # use this to cache previous q-values in ac_true_q
        pass

    def get_policy_prob(self, state):
        # returns vector of policy probabilities
        return softmax(self.param[state[0], state[1]])

    def get_action(self, state):
        # returns an action sampled according to the policy probabilities
        return self.rng.choice(np.arange(0, self.num_actions), p=self.get_policy_prob(state))

    def get_entropy(self, state):
        p = np.array(self.get_policy_prob(state))
        return -np.sum(p*np.log(p))

    def update_reinforce(self, trajectory, step_size, perturb, rew_step_size=None):
        # trajectory is a sequence of transitions for one episode
        # of the form ((s, a, r'), (s', a', r''), ...)
        total_reward = 0
        total_discount = self.discount ** (len(trajectory)-1)

        if self.baseline_type == 'minvar':
            # we compute the estimate of the minvar baseline for the initial state only and use it for all the states in a trajectory
            # this corresponds to the minvar baseline of the reinforce estimator on entire trajectories (not the action-dependent one)
            s_init, _, _ = trajectory[0]
            minvar_baseline = self._estimate_minvar_baseline()

        for transition in reversed(trajectory):
            # gradient
            state, a, r = transition
            s = self._state_index(state)
            onehot = np.zeros(self.num_actions)
            onehot[a] = 1

            total_reward = total_reward * self.discount + r

            if self.baseline_type == 'avg':
                baseline = self.avg_returns[s]
                # update the avg returns
                 #1 / (np.sqrt(self.visit_counts[s[0], s[1]]) + 1)  # could change the step size here
                self.avg_returns[s] += rew_step_size * (total_reward - self.avg_returns[s])
            elif self.baseline_type == 'minvar':
                #baseline = self.env.compute_minvar_baseline(s)
                baseline = minvar_baseline
            else:
                baseline = 0
            self.visit_counts[s] += 1


            self.param[s] += step_size * total_discount * ((total_reward - (baseline + perturb)) * (onehot - self.get_policy_prob(s)))
            # note that this previous step has to be done simultaneously for all states for function approx i
            total_discount /= self.discount

        return

    def update_ac_true_q(self, trajectory, step_size, perturb, rew_step_size=None):
        # this is the actor-critic update with the true q-values
        # trajectory is a sequence of transitions for one episode
        # of the form ((s, a, r'), (s', a', r''), ...)
        total_reward = 0
        total_discount = self.discount ** (len(trajectory) - 1)

        if self.baseline_type == 'minvar':
            # we compute the minvar baseline for all states with the true q-values
            q_values = self._solve_q_values()
            minvar_baselines = self._compute_ac_minvar_baseline(q_values)

        for transition in reversed(trajectory):
            # gradient
            state, a, r = transition
            s = self._state_index(state)
            onehot = np.zeros(self.num_actions)
            onehot[a] = 1

            # total_reward = total_reward * self.discount + r

            if self.baseline_type == 'avg':
                baseline = self.avg_returns[s]
                # update the avg returns
                # 1 / (np.sqrt(self.visit_counts[s[0], s[1]]) + 1)  # could change the step size here
                self.avg_returns[s] += rew_step_size * (total_reward - self.avg_returns[s])
            elif self.baseline_type == 'minvar':
                baseline = minvar_baselines[s]
            else:
                baseline = 0
            self.visit_counts[s] += 1

            self.param[s] += step_size * total_discount * (
                        (q_values[(s[0], s[1], a)] - (baseline + perturb)) * (onehot - self.get_policy_prob(s)))
            # note that this previous step has to be done simultaneously for all states for function approx i

            total_discount /= self.discount

    def _solve_q_values(self):
        # uses dynamic programming to approximately solve for q-values within tolerance
        tolerance = 1e-3

        if self.env.name.lower() == 'fourrooms':
            num_actions = 4
            if self.q_values is None: # initialize array
                self.q_values = 0.5*np.ones((self.env.gridsize[0], self.env.gridsize[1], num_actions))
            q_values = self.q_values  # note this is a reference

            # cache all the possible transitions (assumes self.environment is deterministic)
            transitions = dict()
            for i in range(0, self.env.gridsize[0]):
                for j in range(0, self.env.gridsize[1]):
                    for a in range(0, 4):
                        transitions[(i,j,a)] = self.env._transition(state=(i,j), action=a)
                        # note: transition returns (next_state, reward, done)

            max_change = 999
            while max_change > tolerance:
                # print("change", max_change)
                temp_copy = q_values.copy()
                for i in range(0, self.env.gridsize[0]):
                    for j in range(0, self.env.gridsize[1]):
                        for a in range(0, 4):
                            # print(transitions[(i,j,a)])
                            next_state, reward, done = transitions[(i,j,a)]

                            # next_state = transitions[(i, j, a)]
                            policy_next_state = self.get_policy_prob(next_state)

                            # compute bellman update
                            if done:
                                q_values[(i,j,a)] = reward
                                # print(reward)
                            else:
                                q_values[(i,j,a)] = reward + self.discount * np.sum(q_values[tuple(next_state)] * policy_next_state)

                max_change = np.max(np.abs(q_values - temp_copy))

            return q_values.copy()

        elif self.env.name.lower() == 'gridworld':
            raise AssertionError('Solving q-values Not implemented for gridworld')

    def _compute_ac_minvar_baseline(self, q_values):
        # returns the min-variance baseline for all states
        # requires as input the q-value estimates (could be the true q-values)
        if self.env.name.lower() == 'fourrooms':
            # assumes q_values are given as an array with (state, state, action)
            num_actions = 4
            baselines = np.zeros(tuple(self.env.gridsize), dtype='float')
            for i in range(self.env.gridsize[0]):
                for j in range(self.env.gridsize[1]):
                    policy = self.get_policy_prob([i, j])
                    baselines[(i,j)] = np.sum([self._weight_optimal_baseline(a, policy) * q_values[(i,j,a)] for a in range(num_actions)])
            return baselines

        elif self.env.name.lower() == 'gridworld':
            raise AssertionError('Solving q-values Not implemented for gridworld')

    def _weight_optimal_baseline(self, index, policy_probs):
        denom = 1 - np.sum(np.square(policy_probs))
        num = np.sum(np.square(policy_probs)) - policy_probs[index]**2 + (1-policy_probs[index])**2
        return num * policy_probs[index] / denom

    def _estimate_minvar_baseline(self, num_rollouts=100, importance_sampling=True):
        # uses rollouts to estmate the minimum-variance baseline
        # use importance sampling for better estimates
        # the behaviour policy is set to be uniform random epsilon of the time or else it picks the same action as
        # the target policy 1-epsilon of the time

        behav_policy_epsilon = 0.1

        disc = 0.99
        disc_returns = []
        gradlogprob_sums = []

        env = self.env.copy()
        if env.name == 'gridworld':
            max_steps = 100
        elif env.name == 'fourrooms':
            max_steps = 200

        state = env.reset()

        for i_ep in range(num_rollouts):
            gradlogprob = np.zeros(self.param.shape)  # stores all the gradients

            ep_rewards= []
            # print("ep {}".format(i_ep))
            done = False
            steps = 0
            while not done:
                # print(trajectory)
                prev_state = state

                action = self.get_action(state)

                # compute gradient log prob for current state and add it to the rest
                s = self._state_index(state)
                onehot = np.zeros(self.num_actions)
                onehot[action] = 1
                gradlogprob[s] += onehot - self.get_policy_prob(s)

                state, reward, done = env.step(int(action))

                ep_rewards.append(reward)

                steps += 1
                if steps >= max_steps:
                    break

            ep_disc_return = np.sum([ep_rewards[i] * (disc ** i) for i in range(len(ep_rewards))])
            disc_returns.append(ep_disc_return)
            gradlogprob_sums.append(np.sum(np.square(gradlogprob)))

            # reset
            state = env.reset()

        # check = np.stack([gradlogprob_sums, disc_returns])
        # print(np.round( check[:, check[0,:].argsort()], 2))

        minvar_baseline = np.sum(np.array(disc_returns) * np.array(gradlogprob_sums)) / np.sum(gradlogprob_sums)
        # print(minvar_baseline)
        return minvar_baseline



    def _state_index(self, state):
        ''' Converts a state into the appropriate indices to index the parameters '''
        if self.env.name.lower() in ('gridworld', 'fourrooms'):
            return state[0], state[1]
        elif self.env.name.lower() == 'binomialtree':
            return state[0], state[1]
        else:
            raise AssertionError("invalid env {}".format(self.env.name))

if __name__ == '__main__':
    np.set_printoptions(suppress=True, linewidth=1000)
    print('Running!')
    # gridsize = (5,5)
    step_size = 0.1
    perturb = 0.0
    # num_steps = 1000
    num_episodes = 1000

    env = SimpleMDP.FourRoomsEnv()
    # env = SimpleMDP.BinomialTreeMDP(depth=10)
    agent = PGAgent(num_actions=4, baseline_type='minvar', discount=0.99, env=env)

    # q = agent._solve_q_values()
    # # val = np.mean(q, axis=2)
    # # print(val)
    # b = agent._compute_ac_minvar_baseline(q)
    # print(b)
    # quit()
    state = env.reset()

    start_time = time.perf_counter()
    max_steps = 500
    num_steps_per_ep = []
    returns = []

    for i_ep in range(num_episodes):
        print("ep {}".format(i_ep))
        done = False
        trajectory = []
        steps = 0

        while not done:
            # print(trajectory)
            prev_state = state

            action = agent.get_action(state)
            state, reward, done = env.step(action)
            # print(prev_state, reward, done, state)
            trajectory.append((prev_state, action, reward))
            steps += 1
            if steps >= max_steps:
                break

        # do updates
        print('rew', reward, round(reward * 0.99**steps,3), 'steps', steps, 'goal', state)
        num_steps_per_ep.append(steps)
        # agent.update_reinforce(trajectory, step_size, perturb)
        agent.update_ac_true_q(trajectory, step_size, perturb)
        # reset
        state = env.reset()
        returns.append(reward * 0.99**steps)
        # agent = np.sum([_, _, r for transition in trajectory])

    print(num_steps_per_ep)
    print(returns)
    print((time.perf_counter() - start_time) / 60, 'min')
    pass


