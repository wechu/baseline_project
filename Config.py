#####
# Configuration file for the experiments
#
#####
import time, datetime
import numpy as np
from collections import OrderedDict as OD

class Config:
    def __init__(self):
        # write down all parameter settings here
        self.id = int(time.time())  # use Unix time as a unique id
        now = datetime.datetime.now()
        self.date = '{:02d}_{:02d}_{:02d}_{}'.format(now.month, now.day, now.hour, now.year)
        self.parallelize_runs = False  # use this to split the runs into separate jobs

        self.num_runs = 50
        # self.num_steps = 30000
        self.num_episodes = 3000  # only one of num_eps or num_steps should be used
        self.save_freq = 5
        self.save_dir = 'res'
        self.save_tag = ''  # string to add to the saved file name
        self.output_file = 'log.txt'
        self.logged_values = ['returns', 'discounted_returns', 'action_entropy_trajectory', 'state_visitation_entropy_online', 'state_visitation_entropy_eval']
        # RS, IS; constant0 policy; local detect features

        self.algorithms = ['ac_true_q']
        self.algs_sweep_params = {'ac_true_q': OD([("step_size", [0.001, 0.003, 0.005, 0.01, 0.03, 0.05, 0.1]),
                                                   # ("rew_step_size", [0.0]),
                                                   ("perturb", [-1, -0.5, 0, 0.5, 1])])
                                   }  #

        self.shared_sweep_params = OD([('optimizer', ['SGD']),
                                       ('discount', [0.99]),
                                       ('horizon', [200])])

        self.alg_other_params = {'seed': 123, 'baseline_type': 'minvar', 'use_natural_pg': True}
        # TODO when adding ADAM, think about where to put beta1 and beta2

        self.environments = ['fourrooms']
        self.env_params = [{"seed": None,}
                           ]  # this list matches the order of envs in self.environments,

config = Config()

