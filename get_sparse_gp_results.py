import numpy as np
import pandas as pd
import gpflow
import random
from sklearn.metrics import mean_squared_error
import io
import pickle
import argparse
import os

"""#### CPU cores """
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

recent_df = pd.read_csv('nov-data.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--num_inducing', type=int, default=50, help='number of inducing points')
parser.add_argument('--forecasting', action='store_true', help='forecasting model')
parser.add_argument('--stsvgp', action='store_true', help='forecasting model')

args = parser.parse_args()
M = args.num_inducing
forecasting = args.forecasting
stsvgp = args.stsvgp

if stsvgp:
    results_dir = f'st_svgp_M={M}/'
else:
    results_dir = f'sparseGP_M={M}/'

if forecasting:
    sub_folder = 'forecasting_results'
else:
    sub_folder = 'nowcasting_results'

site_mses = {}
site_vars = {}
total_time_taken = 0.0

# for root, dirs, _ in os.walk(results_dir):
for site_dir in os.listdir(results_dir):
    if os.path.isfile(results_dir + site_dir):
        continue

    path = os.path.join(results_dir, site_dir + "/" + sub_folder)

    if not os.path.exists(path):
        # Some sites do not have results for forecasting
        continue

    f = open(os.path.join(path + "/rmse.txt"), "r")
    site_mses[site_dir] = float(f.read().strip())

    f = open(os.path.join(path + "/avg_uncertainty.txt"), "r")
    site_vars[site_dir] = float(f.read().strip())

    f = open(os.path.join(path + "/total_time_taken.txt"), "r")
    total_time_taken += float(f.read().strip())

    # for dir in dirs:
    #     # for filename in os.listdir(dir):
    #     f = open(os.path.join(root, dir + "/" + sub_folder + "/rmse.txt"), "r")
    #     site_mses[dir] = float(f.read().strip())
    #
    #     f = open(os.path.join(root, dir + "/" + sub_folder + "/avg_uncertainty.txt"), "r")
    #     site_vars[dir] = float(f.read().strip())
    #
    #     f = open(os.path.join(root, dir + "/" + sub_folder + "/total_time_taken.txt"), "r")
    #     total_time_taken += float(f.read().strip())

avg_time_taken = total_time_taken / len(site_mses)
#
# with open(output_folder + '/site_rmses', 'wb') as fp:
#     pickle.dump(site_mses, fp)
os.makedirs(results_dir + sub_folder, exist_ok = True)

import csv
with open(results_dir + sub_folder + '/site_rmses.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(site_mses.items())
    # for key, value in site_mses.items():
    #    writer.writerow([key, value])
    # [fp.write('{0},{1}\n'.format(key, value)) for key, value in site_mses.items()]
with open(results_dir + sub_folder + '/site_vars.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(site_vars.items())

np.savetxt(results_dir + sub_folder + '/avg_time_taken.txt', np.array([avg_time_taken]))

# with open(output_folder + '/site_rmses', 'rb') as fp:
#     sites_mses = pickle.load(fp)
#
# print(sites_mses)
