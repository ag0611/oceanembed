import xarray as xr
import numpy as np
import os

# ── Config ──────────────────────────────────────────────────────────
LAT_MIN, LAT_MAX = 5.0, 30.0
LON_MIN, LON_MAX = 45.0, 105.0

RAW = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\raw'
OUT = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed'
os.makedirs(OUT, exist_ok=True)

# ── Persian Gulf + Red Sea mask ──────────────────────────────────────
MASK_REGIONS = [
    {'lat': (22, 30), 'lon': (48, 57)},   # Persian Gulf
    {'lat': (12, 30), 'lon': (32, 43)},   # Red Sea
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
    return ds.sel({
        lat_var: slice(LAT_MIN, LAT_MAX),
        lon_var: slice(LON_MIN, LON_MAX)
    })

def zscore_norm(ds, var):
    mean = ds[var].mean()
    std  = ds[var].std()
    ds[var + '_norm'] = (ds[var] - mean) / std
    ds[var + '_mean'] = mean
    ds[var + '_std']  = std
    return ds

# ── SST ─────────────────────────────────────────────────────────────
print('Processing SST...')
sst_files = sorted([
    os.path.join(RAW, 'sst', f)
    for f in os.listdir(os.path.join(RAW, 'sst'))
    if f.endswith('.nc')
])
ds_sst = xr.open_mfdataset(sst_files, combine='by_coords')
ds_sst = ds_sst.rename({'lat': 'lat', 'lon': 'lon'})
ds_sst = crop_NIO(ds_sst)
ds_sst = apply_mask(ds_sst)
ds_sst = ds_sst.squeeze('zlev', drop=True)
ds_sst = zscore_norm(ds_sst, 'sst')
ds_sst[['sst', 'sst_norm', 'sst_mean', 'sst_std']].to_netcdf(os.path.join(OUT, 'sst_processed.nc'))
print(f'  SST done — shape: {ds_sst["sst"].shape}')

# ── SSH ─────────────────────────────────────────────────────────────
print('Processing SSH...')
ds_ssh = xr.open_dataset(os.path.join(RAW, 'ssh', 'ssh_NIO_2022.nc'))
ds_ssh = crop_NIO(ds_ssh, lat_var='latitude', lon_var='longitude')
ds_ssh = apply_mask(ds_ssh, lat_var='latitude', lon_var='longitude')
ds_ssh = zscore_norm(ds_ssh, 'sla')
ds_ssh[['sla', 'sla_norm', 'sla_mean', 'sla_std']].to_netcdf(os.path.join(OUT, 'ssh_processed.nc'))
print(f'  SSH done — shape: {ds_ssh["sla"].shape}')

# ── SSS ─────────────────────────────────────────────────────────────
print('Processing SSS...')
ds_sss = xr.open_dataset(os.path.join(RAW, 'sss', 'sss_NIO_2022.nc'))
ds_sss = crop_NIO(ds_sss, lat_var='latitude', lon_var='longitude')
ds_sss = apply_mask(ds_sss, lat_var='latitude', lon_var='longitude')
ds_sss = zscore_norm(ds_sss, 'sos')
ds_sss[['sos', 'sos_norm', 'sos_mean', 'sos_std']].to_netcdf(os.path.join(OUT, 'sss_processed.nc'))
print(f'  SSS done — shape: {ds_sss["sos"].shape}')

print('All preprocessing done!')
