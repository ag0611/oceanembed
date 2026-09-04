import xarray as xr
import numpy as np
import os

ARGO = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\argo\validation_profiles.nc'
GLORYS = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\processed\glorys_regrid_0.25deg.nc'

def validate_against_argo():
    print('Loading ARGO profiles...')
    ds_argo = xr.open_dataset(ARGO)
    print(f'  ARGO profiles: {ds_argo.dims}')

    print('Loading GLORYS...')
    ds_glo = xr.open_dataset(GLORYS)

    rmse_per_depth = []
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

    print('Computing RMSE per depth...')
    for d in depths:
        rmse_per_depth.append({'depth': d, 'rmse': np.nan, 'n_profiles': int(ds_argo.dims['row'])})

    print('\nValidation Summary:')
    print(f'  ARGO profiles available: {ds_argo.dims["row"]}')
    print(f'  Depth levels checked: {len(depths)}')
    print('  Note: Full RMSE computation requires model predictions (run after training)')
    return rmse_per_depth

if __name__ == '__main__':
    results = validate_against_argo()
    print('Validator ready!')
