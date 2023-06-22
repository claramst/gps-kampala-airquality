import bayesnewton
import jax.numpy as jnp
import objax
import numpy as np
import time
import sys
from scipy.cluster.vq import kmeans2
import pandas as pd
import argparse
import os

from jax.lib import xla_bridge

## Our attempts to make a periodic kernel for the ST-SVGP model

"""#### CPU cores """
os.environ["OMP_NUM_THREADS"] = "8"

import tensorflow as tf

tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

df = pd.read_csv('nov-data.csv')

df_outliers = pd.DataFrame()

for site in df.site_id.unique():
  site_df = df[df['site_id']==site]
  Q1 = site_df['pm2_5_calibrated_value'].quantile(0.25)
  Q3 = site_df['pm2_5_calibrated_value'].quantile(0.75)
  IQR = Q3 - Q1
  outlier_df = site_df[((site_df['pm2_5_calibrated_value']<(Q1-1.5*IQR)) | (site_df['pm2_5_calibrated_value']>(Q3+1.5*IQR)))]
  df_outliers = pd.concat([df_outliers, outlier_df], ignore_index=True)


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

print(site_index)
print(sites)
print(test_site)

def datetime_to_epoch(datetime):
    """
        Converts a datetime to a number
        args:
            datatime: is a pandas column
    """
    return datetime.astype('int64')//1e9

def normalise(x, wrt_to):
    return (x - np.mean(wrt_to))/np.std(wrt_to)

def normalise_df(x, wrt_to):
    return (x - np.mean(wrt_to, axis=0))/np.std(wrt_to, axis=0)

def un_normalise_df(x, wrt_to):
    return x* np.std(wrt_to, axis=0) + np.mean(wrt_to, axis=0)

def pad_with_nan_to_make_grid(X, Y):
    #converts data into grid

    N = X.shape[0]

    #construct target grid
    unique_time = np.unique(X[:, 0])
    unique_space = np.unique(X[:, 1:], axis=0)

    Nt = unique_time.shape[0]
    Ns = unique_space.shape[0]

    print('grid size:', N, Nt, Ns, Nt*Ns)

    X_tmp = np.tile(np.expand_dims(unique_space, 0), [Nt, 1, 1])

    time_tmp = np.tile(unique_time, [Ns]).reshape([Nt, Ns], order='F')

    X_tmp = X_tmp.reshape([Nt*Ns, -1])

    time_tmp = time_tmp.reshape([Nt*Ns, 1])

    #X_tmp is the full grid
    X_tmp = np.hstack([time_tmp, X_tmp])

    #Find the indexes in X_tmp that we need to add to X to make a full grid
    _X = np.vstack([X,  X_tmp])
    _Y = np.nan*np.zeros([_X.shape[0], 1])

    _, idx = np.unique(_X, return_index=True, axis=0)
    idx = idx[idx>=N]
    print('unique points: ', idx.shape)

    X_to_add = _X[idx, :]
    Y_to_add = _Y[idx, :]

    X_grid = np.vstack([X, X_to_add])
    Y_grid = np.vstack([Y, Y_to_add])

    #sort for good measure
    _X = np.roll(X_grid, -1, axis=1)
    #sort by time points first
    idx = np.lexsort(_X.T)

    return X_grid[idx], Y_grid[idx]

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[['timestamp', 'pm2_5_calibrated_value', 'site_id', 'site_name', 'latitude', 'longitude']]
df['epoch'] = datetime_to_epoch(df['timestamp'])

df_outliers['timestamp'] = pd.to_datetime(df_outliers['timestamp'])
df_outliers = df_outliers[['timestamp', 'pm2_5_calibrated_value', 'site_id', 'site_name', 'latitude', 'longitude']]

df_outliers['epoch'] = datetime_to_epoch(df_outliers['timestamp'])

X = df[['epoch', 'latitude', 'longitude']].astype('float').to_numpy()
Y = df[['pm2_5_calibrated_value']].to_numpy()

#remove duplicated data
u, unique_idx = np.unique(X, return_index=True, axis=0)
X = X[unique_idx, :]
Y = Y[unique_idx, :]

# For the filtering methods to work we need a full spatio-temporal grid
X_raw, Y_raw = pad_with_nan_to_make_grid(X.copy(), Y.copy())

if forecasting:
    last_day = df[df['timestamp'].astype(str).str.startswith('2021-11-30')]
    test = df.loc[last_day.index]
    test = test[test['site_id'] == test_site]
    if len(test) == 0:
        sys.exit("Site has no readings at forecast test time")

    start_epoch = last_day['epoch'].min()
    end_epoch = last_day['epoch'].max()

    test_latitude = test.latitude.unique()[0]
    test_longitude = test.longitude.unique()[0]

    X_raw_tuples = set(map(tuple, X_raw))
    df_outliers_tuples = set(map(tuple, df_outliers[['epoch', 'latitude', 'longitude']].to_numpy()))
    # Create an array of the same length as X_raw, with True where a row is an outlier and False elsewhere
    outliers_mask = np.array([x in df_outliers_tuples for x in X_raw_tuples])
    last_day_mask = (X_raw[:,0]>=start_epoch) & (X_raw[:,0]<=end_epoch)
    # Masks outliers and the last day. We want to remove these from train set
    train_mask = (outliers_mask | last_day_mask).nonzero()

    # Masks the last day at given site. Should be length 47530 - 24 = 47496
    test_mask = ((X_raw[:,2]!=test_longitude) | (X_raw[:,1]!=test_latitude) | (X_raw[:,0]<start_epoch) | (X_raw[:,0]>end_epoch)).nonzero()

    X_train, Y_train = X_raw.copy(), Y_raw.copy()
    Y_train[train_mask, :] = np.nan #to keep grid structure in X we just mask the testing data in the training set
    # We want to mask last day, and outliers

    X_test, Y_test = X_raw.copy(), Y_raw.copy()
    Y_test[test_mask, :] = np.nan

else:
    test = df[df['site_id']==test_site]
    # train = df_no_outliers[df_no_outliers['site_id'] != test_site]
    #
    test_latitude = test.latitude.unique()[0]
    test_longitude = test.longitude.unique()[0]

    # test_indices = ((X_raw[:,2]==test_longitude) & (X_raw[:,1]==test_latitude)).nonzero()

    X_raw_tuples = set(map(tuple, X_raw))
    df_outliers_tuples = set(map(tuple, df_outliers[['epoch', 'latitude', 'longitude']].to_numpy()))
    outliers_mask = np.array([x in df_outliers_tuples for x in X_raw_tuples])
    site_mask = (X_raw[:,2]!=test_longitude) | (X_raw[:,1]!=test_latitude).nonzero()
    test_site_mask = (X_raw[:,2]==test_longitude) & (X_raw[:,1]==test_latitude)

    train_mask = (outliers_mask | test_site_mask).nonzero()
    test_mask = ((X_raw[:,2]!=test_longitude) | (X_raw[:,1]!=test_latitude)).nonzero()

    X_train, Y_train = X_raw.copy(), Y_raw.copy()
    # We want to mask all outliers, and all readings at test_site
    Y_train[train_mask, :] = np.nan #to keep grid structure in X we just mask the testing data in the training set

    X_test, Y_test = X_raw.copy(), Y_raw.copy()
    # We want to mask all readings not at test_site, but not mask outliers
    Y_test[test_mask, :] = np.nan

X_all = X_raw
Y_all = Y_raw

X_train_norm = normalise_df(X_train, wrt_to=X_train)
X_test_norm = normalise_df(X_test, wrt_to=X_train)
X_all_norm = normalise_df(X_all, wrt_to=X_train)

X = X_train_norm
Y = Y_train
X_t = X_test_norm
Y_t = Y_test

class Periodic(bayesnewton.kernels.QuasiPeriodicMatern12):
  def K(self, X, X2):
    r_per = jnp.pi * jnp.sqrt(jnp.maximum(bayesnewton.utils.square_distance(X, X2), 1e-36)) / self.period
    k_per = jnp.exp(-0.5 * jnp.square(jnp.sin(r_per) / self.lengthscale_periodic))
    r_mat = jnp.sqrt(jnp.maximum(bayesnewton.utils.scaled_squared_euclid_dist(X, X2, self.lengthscale_matern), 1e-36))
    k_mat12 = jnp.exp(-r_mat)
    return self.variance * k_mat12 * k_per


num_z_space = 20

grid = True
print(Y.shape)
print("num data points =", Y.shape[0])

if grid:
    # the gridded approach:
    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)
else:
    # the sequential approach:
    t = X[:, :1]
    R = X[:, 1:]
    t_t = X_t[:, :1]
    R_t = X_t[:, 1:]
Nt = t.shape[0]
print("num time steps =", Nt)
Nr = R.shape[1]
print("num spatial points =", Nr)
N = Y.shape[0] * Y.shape[1] * Y.shape[2]
print("num data points =", N)

var_y = 5.
var_f = 1.
len_time = 0.001
# len_time = 1
len_space = 0.2

sparse = True
opt_z = True  # will be set to False if sparse=False

if sparse:
    z = kmeans2(R[0, ...], num_z_space, minit="points")[0]
else:
    z = R[0, ...]

period = 0.11547016520966591
kern_time = Periodic(variance=var_f, lengthscale_periodic=len_time, period = period)
kern_space0 = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
kern_space1 = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_space)
kern_space = bayesnewton.kernels.Separable([kern_space0, kern_space1])

kern = bayesnewton.kernels.SpatioTemporalKernel(temporal_kernel=kern_time,
                                                spatial_kernel=kern_space,
                                                z=z,
                                                sparse=sparse,
                                                opt_z=opt_z,
                                                conditional='Full')

lik = bayesnewton.likelihoods.Gaussian(variance=var_y)

model = bayesnewton.models.MarkovVariationalGP(kernel=kern, likelihood=lik, X=t, R=R, Y=Y, parallel=None)

lr_adam = 0.01
lr_newton = 1.
iters = 300
opt_hypers = objax.optimizer.Adam(model.vars())
energy = objax.GradValues(model.energy, model.vars())


@objax.Function.with_vars(model.vars() + opt_hypers.vars())
def train_op():
    model.inference(lr=lr_newton)  # perform inference and update variational params
    dE, E = energy()  # compute energy and its gradients w.r.t. hypers
    opt_hypers(lr_adam, dE)
    return E


train_op = objax.Jit(train_op)

t0 = time.time()
for i in range(1, iters + 1):
    loss = train_op()
    print('iter %2d: energy: %1.4f' % (i, loss[0]))
t1 = time.time()
print('optimisation time: %2.2f secs' % (t1-t0))
avg_time_taken = (t1-t0)/iters
total_time_taken = t1 - t0
print('total time taken:')
print(total_time_taken)
print('average iter time: %2.2f secs' % avg_time_taken)

posterior_mean, posterior_var = model.predict_y(X=t_t, R=R_t)
nlpd = model.negative_log_predictive_density(X=t_t, R=R_t, Y=Y_t)
rmse = np.sqrt(np.nanmean((np.squeeze(Y_t) - np.squeeze(posterior_mean))**2))
print('nlpd: %2.3f' % nlpd)
print('rmse: %2.3f' % rmse)

print(test_site)
print(rmse)

avg_uncertainty = np.average(posterior_var)

site_id_formatted = test_site.replace(" ", "")
site_id_formatted = test_site.replace("/", "")

folder_outputs = f'st_svgp_M={M}/' + site_id_formatted + '/'

if forecasting:
    sub_folder = 'forecasting_results'
else:
    sub_folder = 'nowcasting_results'

os.makedirs(folder_outputs, exist_ok = True)
os.makedirs(folder_outputs + sub_folder, exist_ok = True)

np.savetxt(folder_outputs + sub_folder + '/rmse.txt', np.array([rmse]))
np.savetxt(folder_outputs + sub_folder + '/avg_uncertainty.txt', np.array([avg_uncertainty]))
np.savetxt(folder_outputs + sub_folder + '/learned_inducing_points.txt', np.array(kern.z))
np.savetxt(folder_outputs + sub_folder + '/total_time_taken.txt', np.array([total_time_taken]))
np.savetxt(folder_outputs + sub_folder + '/posterior_mean.txt', np.array(posterior_mean))
np.savetxt(folder_outputs + sub_folder + '/test_ys.txt', np.array(np.squeeze(Y_t)))
