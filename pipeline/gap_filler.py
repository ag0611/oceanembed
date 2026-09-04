import xarray as xr
import numpy as np
import os

PROCESSED = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed'

def fill_gaps_rolling(nc_file, var, window=7):
    print(f'Gap filling {var} in {os.path.basename(nc_file)}...')
    ds = xr.open_dataset(nc_file)
    filled = ds[var].rolling(time=window, min_periods=1, center=True).mean()
    ds[var] = ds[var].fillna(filled)
    out = nc_file.replace('.nc', '_gapfilled.nc')
    ds.to_netcdf(out)
    print(f'  Saved: {os.path.basename(out)}')
    return out

if __name__ == '__main__':
    fill_gaps_rolling(os.path.join(PROCESSED, 'sst_processed.nc'), 'sst_norm')
    fill_gaps_rolling(os.path.join(PROCESSED, 'sss_processed.nc'), 'sos_norm')
    print('Gap filling complete!')
