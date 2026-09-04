import numpy as np
import xarray as xr
from svp import compute_svp_profile, mackenzie_svp
from sld import compute_sld
from shadow_zone import compute_shadow_zone
from sofar import compute_sofar_axis

DEPTHS = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]

def process_grid_cell(temp_profile, sal_profile=None, depths=DEPTHS):
    """
    Full acoustic processing for one lat/lon grid cell.
    temp_profile: array of 15 temperatures (C) at standard depths
    sal_profile:  array of 15 salinities (PSU) — defaults to 35 PSU if None
    Returns: dict with svp, sld_depth, sld_vel, sofar_depth, sofar_vel, shadow_zones
    """
    if sal_profile is None:
        sal_profile = [35.0] * len(depths)

    svp = compute_svp_profile(temp_profile, sal_profile, depths)
    sld_depth, sld_vel     = compute_sld(svp, depths)
    sofar_depth, sofar_vel = compute_sofar_axis(svp, depths)
    shadow_zones           = compute_shadow_zone(svp, depths, sld_depth)

    return {
        'svp':          svp.tolist(),
        'sld_depth':    float(sld_depth),
        'sld_vel':      float(sld_vel),
        'sofar_depth':  float(sofar_depth),
        'sofar_vel':    float(sofar_vel),
        'shadow_zones': shadow_zones
    }


def process_glorys_slice(glorys_path, date_idx=0, sal_const=35.0):
    """
    Load one day from GLORYS regridded NetCDF and compute acoustic
    fields for every NIO grid cell.

    glorys_path: path to glorys_regrid_0.25deg.nc
    date_idx:    time index (0 = first day in file)

    Returns: dict of 2D arrays keyed by acoustic variable
    """
    ds = xr.open_dataset(glorys_path)

    # Detect temp variable name (thetao or to)
    temp_var = 'thetao' if 'thetao' in ds else 'to'
    temp_da  = ds[temp_var].isel(time=date_idx)   # (depth, lat, lon)

    # Detect dimension names
    lat_dim  = 'latitude'  if 'latitude'  in ds.dims else 'lat'
    lon_dim  = 'longitude' if 'longitude' in ds.dims else 'lon'
    dep_dim  = 'depth'     if 'depth'     in ds.dims else 'deptht'

    lats = ds[lat_dim].values
    lons = ds[lon_dim].values
    nlat, nlon = len(lats), len(lons)

    # Output grids
    sld_grid   = np.full((nlat, nlon), np.nan)
    sofar_grid = np.full((nlat, nlon), np.nan)
    shadow_floor_grid = np.full((nlat, nlon), np.nan)

    temp_np = temp_da.values  # (15, nlat, nlon)

    for i in range(nlat):
        for j in range(nlon):
            t_col = temp_np[:, i, j]
            if np.any(np.isnan(t_col)):
                continue  # land / masked cell
            sal_col = [sal_const] * len(DEPTHS)
            try:
                result = process_grid_cell(t_col.tolist(), sal_col)
                sld_grid[i, j]         = result['sld_depth']
                sofar_grid[i, j]       = result['sofar_depth']
                shadow_floor_grid[i,j] = result['shadow_zones'][-1]['depth'] \
                                         if result['shadow_zones'] else np.nan
            except Exception:
                continue

    ds.close()
    return {
        'lats':         lats,
        'lons':         lons,
        'sld':          sld_grid,
        'sofar':        sofar_grid,
        'shadow_floor': shadow_floor_grid
    }


if __name__ == '__main__':
    # Quick smoke test with synthetic profile
    temps = [28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    result = process_grid_cell(temps)

    print(f"SLD:   {result['sld_depth']}m  at  {result['sld_vel']:.2f} m/s")
    print(f"SOFAR: {result['sofar_depth']}m  at  {result['sofar_vel']:.2f} m/s")
    print(f"Shadow zones: {len(result['shadow_zones'])} depth levels computed")
    print("acoustic_processor.py ready.")
