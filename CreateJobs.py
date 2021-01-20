# import os
# import numpy as np
# import itertools
# import sys
#
# DIR = "jobs/"
# LINES_PER_FILE = 5
# USE_ID = True
# ## this script will generate a text file of command to hyper-parameter searches


########
from Config import config
from shutil import copyfile
import pickle
import os
import itertools
import collections
import numpy as np

num_lines_per_file = 1 # number of jobs/settings to run per file

base_save_path = '{}/{}/{}_{}/'.format(config.save_dir, config.date, config.id, config.save_tag)
config_path = ''
os.makedirs(base_save_path + config_path, exist_ok=True)
os.makedirs(base_save_path + config_path + 'jobs/', exist_ok=True)


# save a copy of the pickle file in the jobs folder to load
copyfile("Config.py", base_save_path + config_path + "Config.py")
with open(base_save_path + config_path + 'config.pkl', 'wb') as f:
    pickle.dump(config, f)  # used so CCRunExperiment.py can reload the config object


# total_num_runs = 0
file_index = 0   # need to keep track of this so we know how many jobs to run (one for each file)
line_index = 0

for alg in config.algorithms:
    all_sweep_params = collections.OrderedDict(list(config.shared_sweep_params.items()) + list(config.algs_sweep_params[alg].items()))

    num_shared_hyperparam_settings = int(np.prod([len(v) for v in config.shared_sweep_params.values()]))
    num_alg_hyperparam_settings =  int(np.prod([len(v) for v in config.algs_sweep_params[alg].values()]))
    num_hyperparam_settings = num_shared_hyperparam_settings * num_alg_hyperparam_settings

    # total_num_runs += num_hyperparam_settings

    for hyp_index in range(num_hyperparam_settings):
        if config.parallelize_runs:
            for run_index in range(config.num_runs):
                cmd = 'python CCRunExperiment.py --alg {} --hyp_index {} --config_path {} --run_index {}'.format(alg, hyp_index, base_save_path + config_path, run_index)

                with open(base_save_path + config_path + 'jobs/job_' + str(file_index) + '.txt', 'a') as f:
                    f.write(cmd + '\n')

                if (line_index+1) % num_lines_per_file == 0:
                    file_index += 1
                line_index += 1
        else:

            cmd = 'python CCRunExperiment.py --alg {} --hyp_index {} --config_path {}'.format(alg, hyp_index, base_save_path + config_path)

            with open(base_save_path + config_path + 'jobs/job_' + str(file_index) + '.txt', 'a') as f:
                f.write(cmd + '\n')

            if (line_index+1) % num_lines_per_file == 0:
                file_index += 1
            line_index += 1


# print("Total number of jobs", total_num_runs)
print('Total number of files', file_index)
