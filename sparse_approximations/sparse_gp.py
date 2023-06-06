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

"""#### CPU cores """
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

nov_df = pd.read_csv('nov-data.csv')
weather_df = pd.read_csv('weather-data.csv')

nov_df_no_outliers = pd.DataFrame()

for site in nov_df.site_id.unique():
  site_df = nov_df[nov_df['site_id']==site]
  Q1 = site_df['pm2_5_calibrated_value'].quantile(0.25)
  Q3 = site_df['pm2_5_calibrated_value'].quantile(0.75)
  IQR = Q3 - Q1
  final_df = site_df[~((site_df['pm2_5_calibrated_value']<(Q1-1.5*IQR)) | (site_df['pm2_5_calibrated_value']>(Q3+1.5*IQR)))]
  nov_df_no_outliers = pd.concat([nov_df_no_outliers, final_df], ignore_index=True)

df = nov_df
df_no_outliers = nov_df_no_outliers

parser = argparse.ArgumentParser()
parser.add_argument('--site_index', type=int, default=0, help='selects site')
parser.add_argument('--num_inducing', type=int, default=50, help='number of inducing points')
parser.add_argument('--forecasting', action='store_true', help='forecasting model')

args = parser.parse_args()
site_index = args.site_index
M = args.num_inducing
forecasting = args.forecasting

sites = df.site_id.unique()
test_site = sites[site_index]

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

if forecasting:
    last_day = df[df['Day'].astype(str)=='2021-11-30']
    test = df.loc[last_day.index]
    test = test[test['site_id'] == test_site]
    train = df_no_outliers[df_no_outliers['Day'].astype(str) !='2021-11-30']
    if len(test) == 0:
        sys.exit("Site has no readings at forecast test time")
else:
    test = df[df['site_id']==test_site]
    train = df_no_outliers[df_no_outliers['site_id'] != test_site]

"""

#### Mean and standard deviation"""

mean_calibrated_pm2_5 = df['pm2_5_calibrated_value'].mean(axis=0)
std_calibrated_pm2_5 = df['pm2_5_calibrated_value'].std(axis=0)
mean_raw_pm2_5 = df['pm2_5_raw_value'].mean(axis=0)
std_raw_pm2_5 = df['pm2_5_raw_value'].std(axis=0)
mean_latitude = df['latitude'].mean(axis=0)
std_latitude = df['latitude'].std(axis=0)
mean_longitude = df['longitude'].mean(axis=0)
std_longitude = df['longitude'].std(axis=0)

mean_wind_speed = df['windspeed'].mean(axis=0)
std_wind_speed = df['windspeed'].std(axis=0)
mean_wind_direction = df['winddir'].mean(axis=0)
std_wind_direction = df['winddir'].std(axis=0)
mean_temperature = df['temp'].mean(axis=0)
std_temperature = df['temp'].std(axis=0)
mean_precipitation = df['precip'].mean(axis=0)
std_precipitation = df['precip'].std(axis=0)
mean_humidity = df['humidity'].mean(axis=0)
std_humidity = df['humidity'].std(axis=0)
mean_cloud_cover = df['cloudcover'].mean(axis=0)
std_cloud_cover = df['cloudcover'].std(axis=0)

# Kernel
day_period = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[0], lengthscales=[0.14]), period=7)
hour_period = gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(active_dims=[1], lengthscales=[0.04]), period=24)

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

random_indices = np.random.choice(len(train), size=M, replace=False)
Z = X_normalised[[random_indices]].reshape(M, 10)

print(Z.shape)
model = gpflow.models.SGPR((X_normalised, Y_normalised), periodic_kernel,
                           Z)

import time
start = time.time()
opt = gpflow.optimizers.Scipy()
opt.minimize(model.training_loss, model.trainable_variables)
end = time.time()
print("Execution time in seconds: ")
print(end - start)

print("finished training")

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

mse = mean_squared_error(y_mean_unnormalised, testY)
rmse = np.sqrt(mse)
print(test_site)
print(rmse)

site_id_formatted = test_site.replace(" ", "")
site_id_formatted = test_site.replace("/", "")

folder_outputs = f'sparseGP_M={M}/' + site_id_formatted + '/'

if forecasting:
    sub_folder = 'forecasting_results'
else:
    sub_folder = 'nowcasting_results'

os.makedirs(folder_outputs, exist_ok = True)
os.makedirs(folder_outputs + sub_folder, exist_ok = True)

np.savetxt(folder_outputs + sub_folder + '/mse.txt', np.array([rmse]))
np.savetxt(folder_outputs + sub_folder + '/variance.txt', np.array(y_var))
