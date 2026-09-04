import xarray as xr
import numpy as np
import zarr
import os
from scipy.ndimage import zoom

PROCESSED = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed'
ZARR_OUT  = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed\nio_patches.zarr'

PATCH_SIZE = 64
STRIDE     = 32

print('Loading all processed datasets...')
ds_sst  = xr.open_dataset(os.path.join(PROCESSED, 'sst_processed.nc'))
ds_ssh  = xr.open_dataset(os.path.join(PROCESSED, 'ssh_processed.nc'))
ds_sss  = xr.open_dataset(os.path.join(PROCESSED, 'sss_processed.nc'))
ds_cur  = xr.open_dataset(os.path.join(PROCESSED, 'currents_processed.nc'))
ds_wind = xr.open_dataset(os.path.join(PROCESSED, 'winds_daily_processed.nc'))
ds_glo  = xr.open_dataset(os.path.join(PROCESSED, 'glorys_regrid_0.25deg.nc'))
print('  All datasets loaded.')

n_times = len(ds_glo.time.values)
n_lat   = ds_sst.sizes['lat']
n_lon   = ds_sst.sizes['lon']
print(f'  Time steps: {n_times}, Grid: {n_lat} x {n_lon}')

lat_starts = list(range(0, n_lat - PATCH_SIZE + 1, STRIDE))
lon_starts = list(range(0, n_lon - PATCH_SIZE + 1, STRIDE))
n_patches_per_day = len(lat_starts) * len(lon_starts)
total_patches = n_times * n_patches_per_day
print(f'  Patches per day: {n_patches_per_day}, Total: {total_patches}')

store = zarr.open_group(ZARR_OUT, mode='w')
X = store.create_array('X', shape=(total_patches, 5, PATCH_SIZE, PATCH_SIZE), dtype='float32', chunks=(64, 5, 64, 64))
Y = store.create_array('Y', shape=(total_patches, 15, PATCH_SIZE, PATCH_SIZE), dtype='float32', chunks=(64, 15, 64, 64))
T = store.create_array('time_idx', shape=(total_patches,), dtype='int32')

def resize_to(arr, target_shape):
    if arr.shape == target_shape:
        return arr
    factors = (target_shape[0]/arr.shape[0], target_shape[1]/arr.shape[1])
    return zoom(arr, factors, order=1)

print('Extracting patches...')
idx = 0
target = (n_lat, n_lon)

for t in range(n_times):
    if t % 30 == 0:
        print(f'  Day {t+1}/{n_times}...')

    sst_day = ds_sst['sst_norm'].isel(time=t).values
    ssh_day = ds_ssh['sla_norm'].isel(time=t).values
    sss_day = ds_sss['sos_norm'].isel(time=t).squeeze().values
    cur_u   = ds_cur['uo_norm'].isel(time=t).values
    wind_u  = ds_wind['u_wind_norm'].isel(time=t).values
    glo_day = ds_glo['thetao'].isel(time=t).values

    ssh_r  = resize_to(ssh_day, target)
    sss_r  = resize_to(sss_day, target)
    cur_r  = resize_to(cur_u,   target)
    wind_r = resize_to(wind_u,  target)
    glo_r  = np.stack([resize_to(glo_day[d], target) for d in range(15)], axis=0)

    X_day = np.stack([sst_day, ssh_r, sss_r, cur_r, wind_r], axis=0)

    for li in lat_starts:
        for lj in lon_starts:
            X[idx] = X_day[:, li:li+PATCH_SIZE, lj:lj+PATCH_SIZE]
            Y[idx] = glo_r[:, li:li+PATCH_SIZE, lj:lj+PATCH_SIZE]
            T[idx] = t
            idx += 1

print(f'Done! Total patches: {idx}')
print(f'X shape: {X.shape}, Y shape: {Y.shape}')
