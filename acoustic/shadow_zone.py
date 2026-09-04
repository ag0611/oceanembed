import numpy as np
from svp import compute_svp_profile
from sld import compute_sld

def compute_shadow_zone(svp_values, depths, sld_depth, source_depth=50):
    """
    Shadow Zone estimation using Snell's Law below SLD.
    Returns: list of dicts with depth, range_km, angle_deg
    """
    depths = list(depths)
    svp_values = list(svp_values)
    sld_idx = depths.index(sld_depth) if sld_depth in depths else 0
    sld_vel = svp_values[sld_idx]

    below_sld_vel = svp_values[sld_idx:]
    below_depths  = depths[sld_idx:]

    shadow_ranges = []
    for d, v in zip(below_depths, below_sld_vel):
        if v < sld_vel:
            cos_theta = v / sld_vel
            if cos_theta <= 1:
                theta = np.degrees(np.arccos(cos_theta))
                delta_d = d - sld_depth
                if theta > 0:
                    range_km = delta_d / np.tan(np.radians(theta)) / 1000
                    shadow_ranges.append({
                        'depth': d,
                        'range_km': round(range_km, 2),
                        'angle_deg': round(theta, 2)
                    })
    return shadow_ranges

if __name__ == '__main__':
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    temps  = [28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    sals   = [35] * 15
    svp    = compute_svp_profile(temps, sals, depths)
    sld_d, sld_v = compute_sld(svp, depths)
    zones  = compute_shadow_zone(svp, depths, sld_d)
    print(f'SLD at {sld_d}m ({sld_v:.2f} m/s) -- Shadow zones:')
    for z in zones:
        print(f'  Depth {z["depth"]}m -> range {z["range_km"]} km, angle {z["angle_deg"]}deg')
