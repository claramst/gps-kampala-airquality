import bayesnewton
import objax
import numpy as np
import time
import sys
from scipy.cluster.vq import kmeans2
import pandas as pd

from jax.lib import xla_bridge

df = pd.read_csv('nov-data.csv')

nov_df_no_outliers = pd.DataFrame()

for site in nov_df.site_id.unique():
  site_df = nov_df[nov_df['site_id']==site_id]
  Q1 = site_df['pm2_5_calibrated_value'].quantile(0.25)
  Q3 = site_df['pm2_5_calibrated_value'].quantile(0.75)
  IQR = Q3 - Q1
  final_df = site_df[~((site_df['pm2_5_calibrated_value']<(Q1-1.5*IQR)) | (site_df['pm2_5_calibrated_value']>(Q3+1.5*IQR)))]
  nov_df_no_outliers = pd.concat([nov_df_no_outliers, final_df], ignore_index=True)

df = nov_df_no_outliers

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

if forecasting:
    last_day = df[df['Day'].astype(str)=='2021-11-30']
    last_hour = last_day[last_day['IndexTime']==23]
    train = df.drop(last_hour.index)
    test = df.loc[last_hour.index]
    test = test[test['site_id'] == site_id]
    if len(test) == 0:
        sys.exit("Site has no readings at forecast test time")
else:
    test = df[df['site_id']==test_site]
    train = df.drop(test.index)

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

weather_df = pd.read_csv('weather-data.csv')
weather_df['datetime'] = pd.to_datetime(weather_df['datetime'], utc=True)
weather_df = weather_df[['datetime', 'temp', 'dew', 'humidity', 'precip', 'cloudcover', 'windgust', 'windspeed', 'winddir']]

df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[['timestamp', 'pm2_5_calibrated_value', 'site_id', 'site_name', 'latitude', 'longitude']]

df['epoch'] = datetime_to_epoch(df['timestamp'])

df = df.merge(weather_df, left_on='timestamp', right_on='datetime')
df = df.drop(['windgust', 'datetime'], axis=1)

test = df[df['site_id']==test_site]
train = df.drop(test.index)

X = df[['epoch', 'latitude', 'longitude']].astype('float').to_numpy()
Y = df[['pm2_5_calibrated_value']].to_numpy()

#remove duplicated data
u, unique_idx = np.unique(X, return_index=True, axis=0)
X = X[unique_idx, :]
Y = Y[unique_idx, :]

# For the filtering methods to work we need a full spatio-temporal grid
X_raw, Y_raw = pad_with_nan_to_make_grid(X.copy(), Y.copy())

test_latitude = test.latitude.unique()[0]
test_longitude = test.longitude.unique()[0]
test_indices = ((X_raw[:,2]==test_longitude) & (X_raw[:,1]==test_latitude)).nonzero()
train_indices = ((X_raw[:,2]!=test_longitude) | (X_raw[:,1]!=test_latitude)).nonzero()

#Collect training and testing data
X_train, Y_train = X_raw.copy(), Y_raw.copy()
Y_train[test_indices, :] = np.nan #to keep grid structure in X we just mask the testing data in the training set

X_test, Y_test = X_raw.copy(), Y_raw.copy()
Y_test[train_indices, :] = np.nan

X_all = X_raw
Y_all = Y_raw

X_train_norm = normalise_df(X_train, wrt_to=X_train)
X_test_norm = normalise_df(X_test, wrt_to=X_train)
X_all_norm = normalise_df(X_all, wrt_to=X_train)


def train_test_split_indices(N, split=0.8, seed=0):
    np.random.seed(seed)
    rand_index = np.random.permutation(N)

    N_tr =  int(N * split)

    return rand_index[:N_tr], rand_index[N_tr:]

#Normalise Data input
def normalise(x, wrt_to):
    return (x - np.mean(wrt_to))/np.std(wrt_to)

def normalise_df(x, wrt_to):
    return (x - np.mean(wrt_to, axis=0))/np.std(wrt_to, axis=0)

def un_normalise_df(x, wrt_to):
    return x* np.std(wrt_to, axis=0) + np.mean(wrt_to, axis=0)


X = X_train_norm
Y = Y_train
X_t = X_test_norm
Y_t = Y_test

num_z_space = M

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

# kern = bayesnewton.kernels.SpatioTemporalMatern52(variance=var_f,
#                                            lengthscale_time=len_time,
#                                            lengthscale_space=[len_space, len_space],
#                                            z=z,
#                                            sparse=sparse,
#                                            opt_z=opt_z,
#                                            conditional='Full')

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

# if mean_field:
#     model = bayesnewton.models.MarkovVariationalMeanFieldGP(kernel=kern, likelihood=lik, X=t, R=R, Y=Y, parallel=parallel)
# else:
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
print('total time taken: %2.2f secs' % t1 - t0')
print('average iter time: %2.2f secs' % avg_time_taken)

posterior_mean, posterior_var = model.predict_y(X=t_t, R=R_t)
nlpd = model.negative_log_predictive_density(X=t_t, R=R_t, Y=Y_t)
rmse = np.sqrt(np.nanmean((np.squeeze(Y_t) - np.squeeze(posterior_mean))**2))
print('nlpd: %2.3f' % nlpd)
print('rmse: %2.3f' % rmse)

print(test_site)
print(rmse)

site_id_formatted = test_site.replace(" ", "")
site_id_formatted = test_site.replace("/", "")

folder_outputs = f'sparseGP_M={M}/' + site_id_formatted

if forecasting:
    sub_folder = 'forecasting/'
else
    sub_folder = 'nowcasting/'

os.makedirs(folder_outputs, exist_ok = True)

np.savetxt(folder_outputs + sub_folder + '/mse.txt', np.array([rmse]))
np.savetxt(folder_outputs + sub_folder + '/variance.txt', np.array([posterior_var]))
