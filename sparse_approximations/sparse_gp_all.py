import numpy as np
import pandas as pd
import gpflow
import random
from sklearn.metrics import mean_squared_error
import io
import pickle
import argparse
import os
from sys import exit
from scipy.cluster.vq import kmeans2

"""#### CPU cores """
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Training on all sites and testing forecasting on all sites as a whole

df = pd.read_csv('nov-data.csv')
weather_df = pd.read_csv('weather-data.csv')

parser = argparse.ArgumentParser()
parser.add_argument('--num_inducing', type=int, default=50, help='number of inducing points')

args = parser.parse_args()
M = args.num_inducing


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
df = df[['Day', 'Time', 'IndexTime', 'IndexDay', 'timestamp',
'pm2_5_calibrated_value', 'pm2_5_raw_value', 'latitude', 'longitude', 'site_id']]


""" Merge weather data and PM2.5 data """

df['timestamp'] = pd.to_datetime(df['timestamp'])
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], utc=True)

df = df.merge(weather_df, left_on='timestamp', right_on='datetime')
df = df.drop(['windgust', 'datetime'], axis=1)

df_no_outliers = pd.DataFrame()

for site in df.site_id.unique():
  site_df = df[df['site_id']==site]
  Q1 = site_df['pm2_5_calibrated_value'].quantile(0.25)
  Q3 = site_df['pm2_5_calibrated_value'].quantile(0.75)
  IQR = Q3 - Q1
  final_df = site_df[~((site_df['pm2_5_calibrated_value']<(Q1-1.5*IQR)) | (site_df['pm2_5_calibrated_value']>(Q3+1.5*IQR)))]
  df_no_outliers = pd.concat([df_no_outliers, final_df], ignore_index=True)

last_day = df[df['Day'].astype(str)=='2021-11-30']
test = df.loc[last_day.index]
train = df_no_outliers[df_no_outliers['Day'].astype(str) !='2021-11-30']

"""

#### Mean and standard deviation"""

mean_calibrated_pm2_5 = train['pm2_5_calibrated_value'].mean(axis=0)
std_calibrated_pm2_5 = train['pm2_5_calibrated_value'].std(axis=0)
mean_raw_pm2_5 = train['pm2_5_raw_value'].mean(axis=0)
std_raw_pm2_5 = train['pm2_5_raw_value'].std(axis=0)
mean_latitude = train['latitude'].mean(axis=0)
std_latitude = train['latitude'].std(axis=0)
mean_longitude = train['longitude'].mean(axis=0)
std_longitude = train['longitude'].std(axis=0)

mean_wind_speed = train['windspeed'].mean(axis=0)
std_wind_speed = train['windspeed'].std(axis=0)
mean_wind_direction = train['winddir'].mean(axis=0)
std_wind_direction = train['winddir'].std(axis=0)
mean_temperature = train['temp'].mean(axis=0)
std_temperature = train['temp'].std(axis=0)
mean_precipitation = train['precip'].mean(axis=0)
std_precipitation = train['precip'].std(axis=0)
mean_humidity = train['humidity'].mean(axis=0)
std_humidity = train['humidity'].std(axis=0)
mean_cloud_cover = train['cloudcover'].mean(axis=0)
std_cloud_cover = train['cloudcover'].std(axis=0)

# Kernel
day_period = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[0], lengthscales=[0.14]), period=7)
hour_period = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[1], lengthscales=[0.167]), period=24)

rbf1 = gpflow.kernels.SquaredExponential(active_dims=[2], lengthscales=[0.2])
rbf2 = gpflow.kernels.SquaredExponential(active_dims=[3], lengthscales=[0.2])
rbf3 = gpflow.kernels.SquaredExponential(active_dims=[4, 5, 6, 7, 8, 9])

periodic_kernel = day_period + hour_period + (rbf1 * rbf2) + rbf3

gpflow.set_trainable(periodic_kernel.kernels[0].period, False)
gpflow.set_trainable(periodic_kernel.kernels[1].period, False)

print("Made kernel")

Y = train[['pm2_5_calibrated_value']].to_numpy()
Y_normalised = (Y - mean_calibrated_pm2_5) / std_calibrated_pm2_5

X = train[['IndexDay', 'IndexTime', 'latitude', 'longitude',
'windspeed', 'cloudcover', 'winddir', 'temp', 'precip', 'humidity']].astype('float').to_numpy()

X_normalised = X.copy().T
X_normalised[0] /= 7
X_normalised[1] /= 24
X_normalised[2] = (X_normalised[2] - mean_latitude) / std_latitude
X_normalised[3] = (X_normalised[3] - mean_longitude) / std_longitude
X_normalised[4] = (X_normalised[4] - mean_wind_speed) / std_wind_speed
X_normalised[5] = (X_normalised[5] - mean_cloud_cover) / std_cloud_cover
X_normalised[6] = (X_normalised[6] - mean_wind_direction) / std_wind_direction
X_normalised[7] = (X_normalised[7] - mean_temperature) / std_temperature
X_normalised[8] = (X_normalised[8] - mean_precipitation) / std_precipitation
X_normalised[9] = (X_normalised[9] - mean_humidity) / std_humidity
X_normalised = X_normalised.T

Z = kmeans2(X_normalised, M, minit="points")[0]

print(Z.shape)
# model = gpflow.models.SGPR((X_normalised, Y_normalised), periodic_kernel,
#                            Z)

model = gpflow.models.SVGP(periodic_kernel, gpflow.likelihoods.Gaussian(),
                           Z, num_data = len(train))

import time
start = time.time()
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss_closure((X_normalised, Y_normalised)), model.trainable_variables)
end = time.time()
print("Execution time in seconds: ")
print(end - start)
total_time_taken = end - start

print("finished training")

day_kernel_l = periodic_kernel.kernels[0].base_kernel.lengthscales
day_kernel_v = periodic_kernel.kernels[0].base_kernel.variance
hour_kernel_l = periodic_kernel.kernels[1].base_kernel.lengthscales
hour_kernel_v = periodic_kernel.kernels[1].base_kernel.variance
lat_kernel_l = periodic_kernel.kernels[2].kernels[0].lengthscales
lat_kernel_v = periodic_kernel.kernels[2].kernels[0].variance
long_kernel_l = periodic_kernel.kernels[2].kernels[1].lengthscales
long_kernel_v = periodic_kernel.kernels[2].kernels[1].variance
add_kernel_l = periodic_kernel.kernels[3].lengthscales
add_kernel_v = periodic_kernel.kernels[3].variance

testX = test[['IndexDay', 'IndexTime', 'latitude', 'longitude',
'windspeed', 'cloudcover', 'winddir', 'temp', 'precip', 'humidity']].astype('float').to_numpy()

testY = test[['pm2_5_calibrated_value']].to_numpy()

testX_normalised = testX.copy().T
testX_normalised[0] /= 7
testX_normalised[1] /= 24
testX_normalised[2] = (testX_normalised[2] - mean_latitude) / std_latitude
testX_normalised[3] = (testX_normalised[3] - mean_longitude) / std_longitude
testX_normalised[4] = (testX_normalised[4] - mean_wind_speed) / std_wind_speed
testX_normalised[5] = (testX_normalised[5] - mean_cloud_cover) / std_cloud_cover
testX_normalised[6] = (testX_normalised[6] - mean_wind_direction) / std_wind_direction
testX_normalised[7] = (testX_normalised[7] - mean_temperature) / std_temperature
testX_normalised[8] = (testX_normalised[8] - mean_precipitation) / std_precipitation
testX_normalised[9] = (testX_normalised[9] - mean_humidity) / std_humidity
testX_normalised = testX_normalised.T

y_mean, y_var = model.predict_y(testX_normalised)
y_mean_unnormalised = (y_mean * (std_calibrated_pm2_5)) + mean_calibrated_pm2_5

test_copy = test[['IndexDay', 'IndexTime', 'latitude', 'longitude',
'windspeed', 'cloudcover', 'winddir', 'temp', 'precip', 'humidity',
'pm2_5_calibrated_value']]
test_copy['predicted_pm_2_5'] = y_mean_unnormalised
test_copy['uncertainty'] = y_var
test_copy['actual_pm_2_5'] = testY


mse = mean_squared_error(y_mean_unnormalised, testY)
rmse = np.sqrt(mse)
print(rmse)

avg_uncertainty = np.average(y_var)

folder_outputs = f'sparseGP_M={M}/'
sub_folder = 'all_sites'

os.makedirs(folder_outputs, exist_ok = True)
os.makedirs(folder_outputs + sub_folder, exist_ok = True)

test_copy.to_csv(folder_outputs + sub_folder + '/test.csv')
np.savetxt(folder_outputs + sub_folder + '/rmse.txt', np.array([rmse]))
np.savetxt(folder_outputs + sub_folder + '/avg_uncertainty.txt', np.array([avg_uncertainty]))
np.savetxt(folder_outputs + sub_folder + '/learned_inducing_points.txt', np.array(model.inducing_variable.Z))
np.savetxt(folder_outputs + sub_folder + '/total_time_taken.txt', np.array([total_time_taken]))
np.savetxt(folder_outputs + sub_folder + '/test_points.txt', test.astype('float').to_numpy())
np.savetxt(folder_outputs + sub_folder + '/learned_day_l', np.array(day_kernel_l))
np.savetxt(folder_outputs + sub_folder + '/hour_kernel_l', np.array(hour_kernel_l))
np.savetxt(folder_outputs + sub_folder + '/lat_kernel_l', np.array(lat_kernel_l))
np.savetxt(folder_outputs + sub_folder + '/long_kernel_l', np.array(long_kernel_l))
np.savetxt(folder_outputs + sub_folder + '/add_kernel_l', np.array(add_kernel_l))
np.savetxt(folder_outputs + sub_folder + '/learned_day_v', np.array([day_kernel_v]))
np.savetxt(folder_outputs + sub_folder + '/hour_kernel_v', np.array([hour_kernel_v]))
np.savetxt(folder_outputs + sub_folder + '/lat_kernel_v', np.array([lat_kernel_v]))
np.savetxt(folder_outputs + sub_folder + '/long_kernel_v', np.array([long_kernel_v]))
np.savetxt(folder_outputs + sub_folder + '/add_kernel_v', np.array([add_kernel_v]))
