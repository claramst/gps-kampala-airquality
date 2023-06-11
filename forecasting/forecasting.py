import glob
import numpy as np
import pandas as pd
import gpflow
import random
from sklearn.metrics import mean_squared_error
import pickle
import argparse
import os

"""#### CPU cores """
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

df = pd.read_csv('nov-data.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--rbf', action='store_true', help='use rbf kernel')
args = parser.parse_args()
rbf = args.rbf

sites = df.site_id.unique()
print(sites.shape)
print(df.shape)

df = df

def add_times_to_df(df):
    strtime_to_idx = {f"{idx:02d}:00": idx for idx in range(24)}
    days_and_times = np.array([v.split(" ") for v in df.timestamp.values])

    if 'Day' not in df:
        df.insert(0, 'Day', days_and_times[:, 0])
    if 'Time' not in df:
        df.insert(1, 'Time', [time[:-3] for time in days_and_times[:, 1]])
    if 'IndexTime' not in df:
        df.insert(2, 'IndexTime', [strtime_to_idx[time] for time in df.Time.values])
    if 'IndexDay' not in df:
        df['Day'] = pd.to_datetime(df['Day'])
        df.insert(3, 'IndexDay', df['Day'].dt.weekday)

add_times_to_df(df)
df = df[['Day', 'Time', 'IndexTime', 'IndexDay', 'timestamp', 'pm2_5_calibrated_value', 'pm2_5_raw_value', 'latitude', 'longitude', 'site_id']]

print("Number of sites for last day")
last_day = df[df['Day'].astype(str)=='2021-11-30']
print(last_day.site_id.unique().shape)

print("Number of sites for last hour")
last_hour = last_day[last_day['IndexTime']==23]
print(last_hour.site_id.unique().shape)

# train = df.drop(last_hour.index)
train = df.drop(last_day.index)

mean_calibrated_pm2_5 = train['pm2_5_calibrated_value'].mean(axis=0)
std_calibrated_pm2_5 = train['pm2_5_calibrated_value'].std(axis=0)
mean_raw_pm2_5 = train['pm2_5_raw_value'].mean(axis=0)
std_raw_pm2_5 = train['pm2_5_raw_value'].std(axis=0)
mean_latitude = train['latitude'].mean(axis=0)
std_latitude = train['latitude'].std(axis=0)
mean_longitude = train['longitude'].mean(axis=0)
std_longitude = train['longitude'].std(axis=0)

def train_test_forecast_gp(df, site_id, kernel):
    test = df.loc[last_day.index]
    test = test[test['site_id'] == site_id]
    mses = np.zeros(4)

    for i in range(4):
      if len(test) == 0:
        return 0

      if len(train) > 1000:
          rand_train = train.sample(n=1000, random_state=i)
      else:
          rand_train = train

      X = rand_train[['IndexDay', 'IndexTime', 'latitude', 'longitude']].astype('float').to_numpy()
      Y = rand_train[['pm2_5_calibrated_value']].to_numpy()
      X_normalised = X.copy().T
      X_normalised[0] /= 7
      X_normalised[1] /= 24
      X_normalised[2] = (X_normalised[2] - mean_latitude) / std_latitude
      X_normalised[3] = (X_normalised[3] - mean_longitude) / std_longitude
      X_normalised = X_normalised.T

      Y_normalised = (Y - mean_calibrated_pm2_5) / std_calibrated_pm2_5

      model = gpflow.models.GPR(
          (X_normalised, Y_normalised),
          kernel=kernel
      )

      opt = gpflow.optimizers.Scipy()
      opt.minimize(model.training_loss, model.trainable_variables)

      testX = test[['IndexDay', 'IndexTime', 'latitude', 'longitude']].astype('float').to_numpy()
      testY = test[['pm2_5_calibrated_value']].to_numpy()

      testX_normalised = testX.copy().T
      testX_normalised[0] /= 7
      testX_normalised[1] /= 24
      testX_normalised[2] = (testX_normalised[2] - mean_latitude) / std_latitude
      testX_normalised[3] = (testX_normalised[3] - mean_longitude) / std_longitude
      testX_normalised = testX_normalised.T

      y_mean, y_var = model.predict_y(testX_normalised)
      y_mean_unnormalised = (y_mean * (std_calibrated_pm2_5)) + mean_calibrated_pm2_5

      mse = mean_squared_error(y_mean_unnormalised, testY)
      mses[i] = mse
    return np.average(mses)

rbf_kernel = gpflow.kernels.SquaredExponential(lengthscales=[0.14, 0.04, 0.2, 0.2])

day_period = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[0], lengthscales=[0.14]), period=7)
hour_period = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[1], lengthscales=[0.04]), period=24)

rbf1 = gpflow.kernels.SquaredExponential(active_dims=[2], lengthscales=[0.2])
rbf2 = gpflow.kernels.SquaredExponential(active_dims=[3], lengthscales=[0.2])

periodic_kernel = day_period + hour_period + (rbf1 * rbf2)
gpflow.set_trainable(periodic_kernel.kernels[0].period, False)
gpflow.set_trainable(periodic_kernel.kernels[1].period, False)

if rbf:
    kernel = rbf_kernel
else:
    kernel = periodic_kernel

mses = np.zeros(len(sites))
for i in range(0, len(sites)):
    mse = train_test_forecast_gp(df, sites[i], kernel)
    mses[i] = mse

mses = mses[mses != 0]

avg_rmse = np.average(np.sqrt(mses))
max_rmse = np.sqrt(np.max(mses))
min_rmse = np.sqrt(np.min(mses))
print(min_rmse)
print(avg_rmse)
print(max_rmse)

site_rmses = dict(zip(sites, np.sqrt(mses)))

parent_folder = 'forecasting/'
if rbf:
    sub_folder = 'rbf_results/'
else:
    sub_folder = 'periodic_results/'

output_folder = parent_folder + sub_folder
os.makedirs(parent_folder, exist_ok = True)
os.makedirs(output_folder, exist_ok = True)

np.savetxt(output_folder + '/rmses.txt', np.array([min_rmse, avg_rmse, max_rmse]))

import csv
with open(output_folder + '/site_rmses.csv', 'w') as fp:
    writer = csv.writer(fp)
    writer.writerows(site_rmses.items())
