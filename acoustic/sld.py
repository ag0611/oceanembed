import numpy as np
from svp import compute_svp_profile

def compute_sld(svp_values, depths):
    """
    Sonic Layer Depth: depth of maximum sound velocity (positive gradient reversal).
    Returns: (SLD depth in m, max sound velocity in m/s)
    """
    idx_max = np.argmax(svp_values)
    return depths[idx_max], svp_values[idx_max]

if __name__ == '__main__':
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    temps  = [28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    sals   = [35] * 15
    svp = compute_svp_profile(temps, sals, depths)
    sld_depth, sld_vel = compute_sld(svp, depths)
    print(f'Sonic Layer Depth: {sld_depth}m at {sld_vel:.2f} m/s')
