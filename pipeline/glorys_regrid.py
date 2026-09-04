import xarray as xr
import numpy as np
import os

RAW = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\raw'
OUT = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed'

print('Loading GLORYS...')
ds_glorys = xr.open_dataset(os.path.join(RAW, 'glorys', 'glorys_NIO_2022.nc'))
print('  GLORYS dims:', ds_glorys.dims)
print('  GLORYS vars:', list(ds_glorys.data_vars))

# Target 0.25 degree grid
lat_out = np.arange(5.0, 30.25, 0.25)
lon_out = np.arange(45.0, 105.25, 0.25)
ds_out = xr.Dataset({
    'lat': (['lat'], lat_out),
    'lon': (['lon'], lon_out),
})

print('Regridding GLORYS 0.083 -> 0.25 degree (conservative)...')
try:
    import xesmf as xe
    regridder = xe.Regridder(
        ds_glorys.rename({'latitude': 'lat', 'longitude': 'lon'}),
        ds_out,
        method='conservative_normed',
        periodic=False,
        ignore_degenerate=True
    )
    ds_regrid = regridder(ds_glorys.rename({'latitude': 'lat', 'longitude': 'lon'})['thetao'])
    ds_regrid = ds_regrid.to_dataset(name='thetao')
    
    # Standard 15 depth levels
    target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    ds_regrid = ds_regrid.sel(depth=target_depths, method='nearest')
    
    ds_regrid.to_netcdf(os.path.join(OUT, 'glorys_regrid_0.25deg.nc'))
    print(f'  GLORYS regrid done — shape: {ds_regrid["thetao"].shape}')
except ImportError:
    print('xesmf not available — using xarray interp fallback')
    ds_renamed = ds_glorys.rename({'latitude': 'lat', 'longitude': 'lon'})
    target_depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    ds_renamed = ds_renamed.sel(depth=target_depths, method='nearest')
    ds_interp = ds_renamed.interp(lat=lat_out, lon=lon_out, method='linear')
    ds_interp[['thetao']].to_netcdf(os.path.join(OUT, 'glorys_regrid_0.25deg.nc'))
    print(f'  GLORYS interp done — shape: {ds_interp["thetao"].shape}')

print('GLORYS regrid complete!')
