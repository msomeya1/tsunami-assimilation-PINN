### file names ###
fname_lonlatt_data = "data_Tohoku/lonlatt_data_20min.npy"
fname_eta_data = "data_Tohoku/eta_data_20min_SN10.npy"
fname_bathy_data = "share/tohoku.grd"
fname_lonlatt_cols = "share/cols_Tohoku.npy"
fname_station_name = "share/Snet133.txt"
fname_station_lonlat = "share/Snet133_lonlat.txt"

outdir = "result_Tohoku"


### parameters ###
Nepoch1 = 50000 # Adam
Nepoch2 = 50000 # LBFGS
tmax = 20 * 60 # sec. This program assumes tmin = 0
dt = 10 # sec
Nt_data = tmax//dt + 1

layer_size = [3] + [50]*4 + [3]
lossweights = [1, 1, 1, 1] # PDE loss x3 / Data loss 

with open(fname_station_name) as f:
    Stations = [s.rstrip() for s in f.readlines()]
Nst = len(Stations)

print(fname_lonlatt_data, fname_eta_data, outdir)
print("Number of used stations =", Nst)
print("tmax =", tmax)
print("Number of time series =", Nt_data)
print("Layer Size =", layer_size)
print("Loss Weights =", lossweights)



### import libraries ###
import os
import numpy as np
from scipy import interpolate
from netCDF4 import Dataset
import matplotlib.pyplot as plt
import seaborn as sns
import deepxde as dde

import torch 
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device = ", device)
print("# of GPU = ", torch.cuda.device_count())

os.makedirs(outdir, exist_ok=True)
os.makedirs(outdir + "/model", exist_ok=True)
os.makedirs(outdir + "/estimated_waveforms", exist_ok=True)
os.makedirs(outdir + "/eta_estimated", exist_ok=True)
os.makedirs(outdir + "/M_estimated", exist_ok=True)
os.makedirs(outdir + "/N_estimated", exist_ok=True)





### prepare D(x,y) ###
# read bathymetry data
bathy = Dataset(fname_bathy_data,'r')
lon_min, lon_max = bathy.variables['x_range'][:]
lat_min, lat_max = bathy.variables['y_range'][:]
Nlon, Nlat = bathy.variables['dimension'][:]
D = bathy.variables['z'][:] 

print("lon_min, lon_max =", lon_min, lon_max)
print("lat_min, lat_max =", lat_min, lat_max)
print("Nlon, Nlat =", Nlon, Nlat)

D = D.reshape((Nlat, Nlon))
D = np.flipud(D)
D[D<=0.0] = 0.0 # land (D<=0) is treated as D=0


# lon/lat in radian
# note: in this program variables with the first letter capitalized (Lon, Lat) are measured by radian. 
Lon_min = np.deg2rad(lon_min)
Lon_max = np.deg2rad(lon_max)
Lat_min = np.deg2rad(lat_min)
Lat_max = np.deg2rad(lat_max)
print(Lon_min, Lon_max)
print(Lat_min, Lat_max)


delta = np.max([Lon_max-Lon_min, Lat_max-Lat_min])

Lon_bathy = (np.linspace(Lon_min, Lon_max, Nlon) - Lon_min) / delta
Lat_bathy = (np.linspace(Lat_min, Lat_max, Nlat) - Lat_min) / delta

print(Lon_bathy)
print(Lat_bathy)

Lon0, Lon1 = Lon_bathy.min(), Lon_bathy.max()
Lat0, Lat1 = Lat_bathy.min(), Lat_bathy.max()
print("Lon0, Lon1 =", Lon0, Lon1)
print("Lat0, Lat1 =", Lat0, Lat1)


### LLW equation ###
g = 9.81
R = 6378 * 1e3
Omega = 7.292e-5 * tmax
print("Omega (non-dimension) =", Omega)


def c2_func(x):
    c2 = g * D.T / (R*delta/tmax)**2
    spline = interpolate.RectBivariateSpline(Lon_bathy, Lat_bathy, c2)
    return spline.ev(x[:, 0:1], x[:, 1:2])

# input = [Lon, lat, t] 
# output = [eta, M, N]
def LLW(input, output, c2):
    Lat = Lat_min + delta * input[:, 1:2]
    cos_Lat = torch.cos(Lat)
    sin_Lat = torch.sin(Lat)

    M = output[:, 1:2]
    N = output[:, 2:3]

    eta_Lon = dde.grad.jacobian(output, input, i=0, j=0) # eta_Lon means ∂(eta)/∂(Lon)
    eta_Lat = dde.grad.jacobian(output, input, i=0, j=1)
    eta_t = dde.grad.jacobian(output, input, i=0, j=2)

    M_Lon = dde.grad.jacobian(output, input, i=1, j=0)
    M_t = dde.grad.jacobian(output, input, i=1, j=2)

    N_Lat = dde.grad.jacobian(output, input, i=2, j=1)
    N_t = dde.grad.jacobian(output, input, i=2, j=2)
    
    return [
        eta_t + (M_Lon + N_Lat * cos_Lat - delta * N * sin_Lat) / cos_Lat,
        M_t + c2 * eta_Lon / cos_Lat - 2 * Omega * sin_Lat * N,
        N_t + c2 * eta_Lat + 2 * Omega * sin_Lat * M
    ]


### data ###


def normalize_lonlatt(lonlatt):
    Lon = (np.deg2rad(lonlatt[:, 0]) - Lon_min) / delta
    Lat = (np.deg2rad(lonlatt[:, 1]) - Lat_min) / delta
    t = lonlatt[:, 2] / tmax

    return np.vstack((np.ravel(Lon), np.ravel(Lat), np.ravel(t))).T


lonlatt_data = np.load(fname_lonlatt_data)
print(lonlatt_data.shape)
print(lonlatt_data)
lonlatt_data = normalize_lonlatt(lonlatt_data)
print(lonlatt_data)

eta_data = np.load(fname_eta_data)
print(eta_data.shape)
print(eta_data)
eta_max, eta_min = eta_data.max(), eta_data.min()
print(eta_max, eta_min)
eta_data = (eta_data - eta_min) / (eta_max - eta_min)
print(eta_data)

eta_observation = dde.icbc.PointSetBC(lonlatt_data, eta_data, component = 0)


### collocation points ###
lonlatt_cols = np.load(fname_lonlatt_cols)
print(lonlatt_cols.shape)
print(lonlatt_cols)
lonlatt_cols = normalize_lonlatt(lonlatt_cols)
print(lonlatt_cols)


# define model 
dde.config.set_random_seed(123)
geom = dde.geometry.Rectangle([Lon0, Lat0], [Lon1, Lat1])
timedomain = dde.geometry.TimeDomain(0, 1)
geomtime = dde.geometry.GeometryXTime(geom, timedomain)
PDE = dde.data.TimePDE(
    geomtime, LLW, [eta_observation],
    num_domain=0,
    auxiliary_var_function=c2_func,
    anchors=lonlatt_cols
)
NN = dde.nn.FNN(
  layer_size, "tanh", "Glorot uniform"
)

def HardIC_MN(input, output):
    t = input[:, 2:3]
    eta = output[:, 0:1]
    M = output[:, 1:2]
    N = output[:, 2:3]
    return torch.cat((eta, t * M, t * N), dim=1)

NN.apply_output_transform(HardIC_MN)




### train model ###
model = dde.Model(PDE, NN)

# Adam
model.compile(
    optimizer="adam", lr=0.001, 
    loss="MSE",
    loss_weights=lossweights
)
losshistory, train_state = model.train(
    model_save_path=outdir+"/model/model",
    iterations=Nepoch1,
    display_every=50
)
dde.utils.external.save_loss_history(losshistory, fname=outdir+"/loss_history.txt")

#LBFGS
dde.optimizers.set_LBFGS_options(maxiter=Nepoch2)
model.compile(
    optimizer="L-BFGS-B",
    loss="MSE",
    loss_weights=lossweights
)
losshistory, train_state = model.train(
    model_save_path=outdir+"/model/model",
    display_every=50,
)
dde.utils.external.save_loss_history(losshistory, fname=outdir+"/loss_history.txt")


### plot loss history ###
loss_history = np.loadtxt(outdir+"/loss_history.txt")

steps = loss_history[:,0]
PDE_loss = loss_history[:,1] + loss_history[:,2] + loss_history[:,3] 
data_loss = loss_history[:,4]
total_loss = PDE_loss + data_loss

sns.set(font_scale=1.4)
plt.rcParams["font.size"] = 20  
fig, ax = plt.subplots(figsize=(6,5), dpi=300)
ax.semilogy(steps, PDE_loss, lw=3, label="PDE loss")
ax.semilogy(steps, data_loss, lw=3, label="Data loss")
ax.semilogy(steps, total_loss, lw=3, label="Total loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.legend()
plt.savefig(outdir+"/loss_history.png", bbox_inches='tight')







### caculate VR_OBPGs ###
final_data_loss = data_loss[-1]
eta_data_squared = np.linalg.norm(eta_data, ord=2)**2 / len(eta_data)
VR = 100*(1-final_data_loss/eta_data_squared)
print("VR of OBPG waveforms (%) =", VR)



### estimation ###

def save_GMTgrd(filename, variable, lon_min, lon_max, lat_min, lat_max, netCDF_format='NETCDF3_CLASSIC'):
    """Save 2D array (lat, lot) as a netCDF file.

    Args:
        filename: File name.
        variable: 2D array, size = (Nlat, Nlon). First dimension (lat) will be inverted and then saved. 
        lon_min,lon_max: Longitude range.
        lat_min,lat_max: Latitude range.
        t_min,t_max: Time range.
        netCDF_format: One of `'NETCDF4'`, `'NETCDF4_CLASSIC'`, `'NETCDF3_CLASSIC'`, `'NETCDF3_64BIT_OFFSET'` or `'NETCDF3_64BIT_DAT'`. \
        Default is `'NETCDF3_CLASSIC'` so that it has the same format as the GMT's grid file. \
        See https://unidata.github.io/netcdf4-python/#netCDF4.Dataset for more details.
    """

    nlat, nlon = variable.shape
    dlon = (lon_max - lon_min) / (nlon - 1)
    dlat = (lat_max - lat_min) / (nlat - 1)

    with Dataset(filename, 'w', format=netCDF_format) as nc: 

        # define dimensions
        nc.createDimension("side", 2)
        nc.createDimension("xysize", nlat * nlon)

        # define variables
        x_range = nc.createVariable("x_range", "f8", "side")
        y_range = nc.createVariable("y_range", "f8", "side")
        z_range = nc.createVariable("z_range", "f8", "side")
        spacing = nc.createVariable("spacing", "f8", "side")
        dimension = nc.createVariable("dimension", "i4", "side")
        z = nc.createVariable("z", "f4", "xysize")

        # create variables
        x_range[:] = np.array([lon_min, lon_max])
        y_range[:] = np.array([lat_min, lat_max])
        z_range[:] = np.array([np.min(variable), np.max(variable)])
        spacing[:] = np.array([dlon, dlat])
        dimension[:] = np.array([nlon, nlat])
        z[:] = variable[::-1,:].flatten()

        # attributes
        x_range.units = "x"
        y_range.units = "y"
        z_range.units = "z"
        z.scale_factor = 1.0
        z.add_offset = 0.0
        z.node_offset = 0

        nc.title = ""
        nc.source = "Created by `save_GMTgrd` (user-defined function of Python)"



# output every 1 min.
Nt_est = tmax // 60

print("Output wavefield estimation")
for i in range(Nt_est+1):
    t = i / Nt_est
    Lat_est, t_est, Lon_est = np.meshgrid(Lat_bathy, np.array([t]), Lon_bathy)
    LonLatt_est = np.vstack((np.ravel(Lon_est), np.ravel(Lat_est), np.ravel(t_est))).T

    # eta
    eta_pred = model.predict(LonLatt_est)[:, 0:1]
    eta_pred = eta_pred.reshape(Nlat, Nlon) * (eta_max-eta_min) + eta_min # denormalized
    save_GMTgrd(outdir+f"/eta_estimated/eta_estimated_{i}.grd", eta_pred, lon_min, lon_max, lat_min, lat_max)

    # M
    M_pred = model.predict(LonLatt_est)[:, 1:2]
    M_pred = M_pred.reshape(Nlat, Nlon) * (eta_max-eta_min) * R * delta / tmax # denormalized
    save_GMTgrd(outdir+f"/M_estimated/M_estimated_{i}.grd", M_pred, lon_min, lon_max, lat_min, lat_max)

    # N
    N_pred = model.predict(LonLatt_est)[:, 2:3]
    N_pred = N_pred.reshape(Nlat, Nlon) * (eta_max-eta_min) * R * delta / tmax # denormalized
    save_GMTgrd(outdir+f"/N_estimated/N_estimated_{i}.grd", N_pred, lon_min, lon_max, lat_min, lat_max)




# waveforms
lonlat_st = np.loadtxt(fname_station_lonlat)
Lon_st = (np.deg2rad(lonlat_st[:, 0]) - Lon_min) / delta
Lat_st = (np.deg2rad(lonlat_st[:, 1]) - Lat_min) / delta
ts = np.linspace(0, 1, Nt_data)

print("Output waveform estimation")
for ist in range(len(Stations)):
    station = Stations[ist]
    Lat_est, t_est, Lon_est = np.meshgrid(np.array([Lat_st[ist]]), ts, np.array([Lon_st[ist]]))
    LonLatt_est = np.vstack((np.ravel(Lon_est), np.ravel(Lat_est), np.ravel(t_est))).T

    pred = model.predict(LonLatt_est)[:, 0:1]
    pred = pred.reshape(Nt_data) * (eta_max-eta_min) + eta_min
    np.savetxt(outdir + f"/estimated_waveforms/{station}.txt", pred)

print("Finish.")
