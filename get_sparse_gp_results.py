import numpy as np
import pandas as pd
import gpflow
import random
from sklearn.metrics import mean_squared_error
import io
import pickle
import argparse
import os
# dirname = "LatestData/"
# csvfiles = glob.glob(f"{dirname}/*")

"""#### CPU cores """
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

recent_df = pd.read_csv('nov-data.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--num_inducing', type=int, default=50, help='number of inducing points')

args = parser.parse_args()
M = args.num_inducing


results_dir = f'sparseGP_M={M}'
site_mses = {}

for root, dirs, _ in os.walk(results_dir):
    for dir in dirs:
        # for filename in os.listdir(dir):
        f = open(os.path.join(root, dir + "/mse.txt"), "r")
        site_mses[dir] = float(f.read().strip())

output_folder = f'sparseGP_M={M}/'
os.makedirs(output_folder, exist_ok = True)
#
# with open(output_folder + '/site_rmses', 'wb') as fp:
#     pickle.dump(site_mses, fp)

import csv
with open(output_folder + '/site_rmses.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(site_mses.items())
    # for key, value in site_mses.items():
    #    writer.writerow([key, value])
    # [fp.write('{0},{1}\n'.format(key, value)) for key, value in site_mses.items()]


# with open(output_folder + '/site_rmses', 'rb') as fp:
#     sites_mses = pickle.load(fp)
#
# print(sites_mses)
