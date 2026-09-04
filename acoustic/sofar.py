import numpy as np
from svp import compute_svp_profile

def compute_sofar_axis(svp_values, depths):
    """
    SOFAR (Sound Fixing and Ranging) channel axis:
    depth of minimum sound velocity.
    Returns: (SOFAR depth in m, min sound velocity in m/s)
    """
    idx_min = np.argmin(svp_values)
    return depths[idx_min], svp_values[idx_min]

if __name__ == '__main__':
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    temps  = [28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    sals   = [35] * 15
    svp    = compute_svp_profile(temps, sals, depths)
    sofar_depth, sofar_vel = compute_sofar_axis(svp, depths)
    print(f'SOFAR axis: {sofar_depth}m at {sofar_vel:.2f} m/s')
