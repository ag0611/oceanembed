import xarray as xr
import os

OUT = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed'

print('Converting winds to daily mean...')
ds_wind = xr.open_dataset(os.path.join(OUT, 'winds_processed.nc'))
ds_daily = ds_wind.resample(time='1D').mean()
ds_daily = ds_daily.rename({
    'eastward_wind': 'u_wind',
    'eastward_wind_norm': 'u_wind_norm',
    'northward_wind': 'v_wind',
    'northward_wind_norm': 'v_wind_norm',
})
ds_daily.to_netcdf(os.path.join(OUT, 'winds_daily_processed.nc'))
print(f'  Winds daily done — shape: {ds_daily["u_wind"].shape}')
print('Done!')
