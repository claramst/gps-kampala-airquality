import bayesnewton
import objax
import numpy as np
import time
import sys
from scipy.cluster.vq import kmeans2
import pandas as pd
import argparse
import os
import geopandas as gpd
from shapely.geometry import Point

from jax.lib import xla_bridge

df = pd.read_csv('nov-data.csv')

df_no_outliers = pd.DataFrame()

for site in df.site_id.unique():
  site_df = df[df['site_id']==site]
  Q1 = site_df['pm2_5_calibrated_value'].quantile(0.25)
  Q3 = site_df['pm2_5_calibrated_value'].quantile(0.75)
  IQR = Q3 - Q1
  final_df = site_df[~((site_df['pm2_5_calibrated_value']<(Q1-1.5*IQR)) | (site_df['pm2_5_calibrated_value']>(Q3+1.5*IQR)))]
  df_no_outliers = pd.concat([df_no_outliers, final_df], ignore_index=True)

parser = argparse.ArgumentParser()

args = parser.parse_args()

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

def train_ST_SVGP_model(train_sites, test_sites, X_raw, Y_raw):
    # train = pd.merge(df_no_outliers, train_sites, how ='inner', on =['latitude', 'longitude'])
    # test = pd.merge(df, test_sites, how ='inner', on =['latitude', 'longitude'])

    X_raw_tuples = set([tuple(x) for x in X_raw[:, 1:]])
    train_sites_tuples = set([tuple(x) for x in train_sites.to_numpy()])
    common_tuples_train = X_raw_tuples.intersection(train_sites_tuples)
    train_indices = [i for i, v in enumerate(X_raw[:, 1:]) if tuple(v) in common_tuples_train]

    test_sites_tuples = set([tuple(x) for x in test_sites.to_numpy()])
    common_tuples_test = X_raw_tuples.intersection(test_sites_tuples)
    test_indices = [i for i, v in enumerate(X_raw[:, 1:]) if tuple(v) in common_tuples_test]

    X_train, Y_train = X_raw.copy(), Y_raw.copy()
    Y_train[test_indices, :] = np.nan #to keep grid structure in X we just mask the testing data in the training set

    X_test, Y_test = X_raw.copy(), Y_raw.copy()
    Y_test[train_indices, :] = np.nan

    X_all = X_raw
    Y_all = Y_raw

    X_train_norm = normalise_df(X_train, wrt_to=X_train)
    X_test_norm = normalise_df(X_test, wrt_to=X_train)
    X_all_norm = normalise_df(X_all, wrt_to=X_train)

    X = X_train_norm
    Y = Y_train
    X_t = X_test_norm
    Y_t = Y_test

    num_z_space = 66

    print(Y.shape)
    print("num data points =", Y.shape[0])

    t, R, Y = bayesnewton.utils.create_spatiotemporal_grid(X, Y)
    t_t, R_t, Y_t = bayesnewton.utils.create_spatiotemporal_grid(X_t, Y_t)

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

    kern_time = bayesnewton.kernels.Matern32(variance=var_f, lengthscale=len_time)
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
    # print('optimisation time: %2.2f secs' % (t1-t0))
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

    print(rmse)
    return model, rmse

# Krause functions
def naiveSensorPlacement(cov, k, S, U):
    """ This is an implementation of the first approximation method suggested in
        the 'Near-Optimal Sensor Placement' paper.
        Input:
        - cov: covariance matrix
        - k: number of Sensors to be placed
        - V: indices of all position
        - S: indices of all possible sensor positions
        - U: indices of all impossible sensor positions
    """
    # print('Algorithm is starting for subdomain', subdomain, flush=True)
    A = []
    V = np.concatenate((S, U), axis=0)
    for j in range(k):
        S_A = np.setdiff1d(S, A).astype(int)
        delta = np.array([])
        for y in S_A:
            AHat = np.setdiff1d(V, np.append(A, [y]))
            delta = np.append(delta, conditionalVariance(cov, y, A) / \
                                      conditionalVariance(cov, y, AHat))
        y_star = S_A[np.argmax(delta)]
        A = np.append(A, y_star).astype(int)
    return A


def conditionalVariance(cov, y, A):
    """ This method calculates the conditional variance of y given A. """
    cov_yy = cov[y, y]
    cov_yA = cov[y, A]
    inv_cov_AA = np.linalg.inv(cov[np.ix_(A, A)])
    cov_Ay = cov[A, y]
    var = cov_yy - ((cov_yA @ inv_cov_AA) @ cov_Ay)

    return var

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[['timestamp', 'pm2_5_calibrated_value', 'site_id', 'site_name', 'latitude', 'longitude']]
df['epoch'] = datetime_to_epoch(df['timestamp'])

X = df[['epoch', 'latitude', 'longitude']].astype('float').to_numpy()
Y = df[['pm2_5_calibrated_value']].to_numpy()

#remove duplicated data
u, unique_idx = np.unique(X, return_index=True, axis=0)
X = X[unique_idx, :]
Y = Y[unique_idx, :]

# For the filtering methods to work we need a full spatio-temporal grid
X_raw, Y_raw = pad_with_nan_to_make_grid(X.copy(), Y.copy())

sites = df[['latitude', 'longitude']].drop_duplicates()

# Select train, candidate and test sites
train_sites = sites.sample(n=40, random_state = 1)
candidate_sites = sites.drop(train_sites.index).sample(n = 15, random_state = 1)
test_sites = sites.drop(train_sites.index)
test_sites = test_sites.drop(candidate_sites.index)

mean_latitude = train_sites['latitude'].mean(axis=0)
std_latitude = train_sites['latitude'].std(axis=0)
mean_longitude = train_sites['longitude'].mean(axis=0)
std_longitude = train_sites['longitude'].std(axis=0)

# avg_uncertainty = np.average(posterior_var)
model, rmse = train_ST_SVGP_model(train_sites, test_sites, X_raw, Y_raw)

folder_outputs = 'krause_st_svgp'
os.makedirs(folder_outputs, exist_ok = True)
np.savetxt(folder_outputs + '/initial_rmse.txt', np.array([rmse]))
train_sites.to_csv(folder_outputs + '/train_sites.csv')
candidate_sites.to_csv(folder_outputs + 'candidate_sites.csv')
test_sites.to_csv(folder_outputs + '/test_sites.csv')


# GENERATE U
lat_range = np.linspace(0.21, 0.41, 15)
lon_range = np.linspace(32.44, 32.7, 15)

lat_points, lon_points = np.meshgrid(lat_range, lon_range)

lat_points = np.array([lat_points.flatten()]).T
lon_points = np.array([lon_points.flatten()]).T

U = np.concatenate((lat_points, lon_points), axis=1)

kampala_outline = gpd.read_file('subcountieskampala.geojson')
kampala_outline = kampala_outline.to_crs("EPSG:4326")

U_df = pd.DataFrame(U, columns=["latitude", "longitude"])

gdf_points = gpd.GeoDataFrame(
    U_df,
    geometry=gpd.points_from_xy(U[:, 1], U[:, 0]),
)

# Make sure both GeoDataFrames are using the same coordinate reference system
gdf_points.set_crs(kampala_outline.crs, inplace=True)

# Filter points that are within the Kampala border using spatial join
points_in_kampala_gdf = gpd.sjoin(gdf_points, kampala_outline, op="within")

U = points_in_kampala_gdf[['latitude', 'longitude']].to_numpy()
U_norm = normalise(U, U)

train_sites_norm = normalise(train_sites.to_numpy(), sites.to_numpy())
candidate_sites_norm = normalise(candidate_sites.to_numpy(), sites.to_numpy())

S_norm = np.concatenate((train_sites_norm, candidate_sites_norm))

V = np.concatenate((S_norm, U_norm), axis=0)

cov = model.kernel(V, V)
# cov = np.random.rand(113, 113)

# 40 training sites. 15 candidate sites. 11 test sites.
# 58 points in across kampala Kampala.
S_indices = np.arange(0, 55, 1, dtype=int)
U_indices = np.arange(55, 113, 1, dtype=int)

# A = naiveSensorPlacement(cov, 40, S_norm, U_norm)
A = naiveSensorPlacement(cov, 40, S_indices, U_indices)

new_sensor_locations = V[A]
new_sensor_locations[0] = (new_sensor_locations[0] * std_latitude) + mean_latitude
new_sensor_locations[1] = (new_sensor_locations[1] * std_longitude) + mean_longitude

np.savetxt(folder_outputs + '/new_sensor_locations.txt', np.array(new_sensor_locations))

new_sensor_locations_df = pd.DataFrame(new_sensor_locations, columns=["latitude", "longitude"])

# new_sensor_locations_df = un_normalise_df(new_sensor_locations_norm_df, train_sites)
print(new_sensor_locations_df)
model, rmse = train_ST_SVGP_model(new_sensor_locations_df, test_sites, X_raw, Y_raw)

folder_outputs = 'krause_st_svgp'
np.savetxt(folder_outputs + '/post_krause_rmse.txt', np.array([rmse]))
new_sensor_locations_df.to_csv(folder_outputs + '/optimal_sensor_locations.csv')
