import xarray as xr
import numpy as np
import os

LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0
RAW = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\raw'
OUT = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed'

MASK_REGIONS = [
    {'lat': (22, 30), 'lon': (48, 57)},
    {'lat': (12, 30), 'lon': (32, 43)},
]

def apply_mask(ds, lat_var='lat', lon_var='lon'):
    mask = xr.ones_like(ds[lat_var], dtype=bool)
    for region in MASK_REGIONS:
        m = (
            (ds[lat_var] >= region['lat'][0]) & (ds[lat_var] <= region['lat'][1]) &
            (ds[lon_var] >= region['lon'][0]) & (ds[lon_var] <= region['lon'][1])
        )
        mask = mask & ~m
    return ds.where(mask)

def crop_NIO(ds, lat_var='lat', lon_var='lon'):
    return ds.sel({lat_var: slice(LAT_MIN, LAT_MAX), lon_var: slice(LON_MIN, LON_MAX)})

def zscore_norm(ds, var):
    mean = ds[var].mean()
    std  = ds[var].std()
    ds[var + '_norm'] = (ds[var] - mean) / std
    ds[var + '_mean'] = mean
    ds[var + '_std']  = std
    return ds

# ── CURRENTS ────────────────────────────────────────────────────────
print('Processing Currents...')
ds_cur = xr.open_dataset(os.path.join(RAW, 'currents', 'currents_NIO_2022.nc'))
print('  Currents dims:', ds_cur.dims)
print('  Currents vars:', list(ds_cur.data_vars))
ds_cur = crop_NIO(ds_cur, lat_var='latitude', lon_var='longitude')
ds_cur = apply_mask(ds_cur, lat_var='latitude', lon_var='longitude')
ds_cur = ds_cur.isel(depth=0, drop=True)
ds_cur = zscore_norm(ds_cur, 'uo')
ds_cur = zscore_norm(ds_cur, 'vo')
ds_cur[['uo', 'uo_norm', 'uo_mean', 'uo_std',
        'vo', 'vo_norm', 'vo_mean', 'vo_std']].to_netcdf(os.path.join(OUT, 'currents_processed.nc'))
print(f'  Currents done — shape: {ds_cur["uo"].shape}')

# ── WINDS ────────────────────────────────────────────────────────────
print('Processing Winds...')
ds_wind = xr.open_dataset(os.path.join(RAW, 'winds', 'winds_NIO_2022.nc'))
print('  Winds dims:', ds_wind.dims)
print('  Winds vars:', list(ds_wind.data_vars))
ds_wind = crop_NIO(ds_wind, lat_var='latitude', lon_var='longitude')
ds_wind = apply_mask(ds_wind, lat_var='latitude', lon_var='longitude')
ds_wind = zscore_norm(ds_wind, 'eastward_wind')
ds_wind = zscore_norm(ds_wind, 'northward_wind')
ds_wind[['eastward_wind', 'eastward_wind_norm', 'eastward_wind_mean', 'eastward_wind_std',
         'northward_wind', 'northward_wind_norm', 'northward_wind_mean', 'northward_wind_std']].to_netcdf(os.path.join(OUT, 'winds_processed.nc'))
print(f'  Winds done — shape: {ds_wind["eastward_wind"].shape}')

print('Currents + Winds preprocessing done!')
