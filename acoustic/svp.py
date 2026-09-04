import numpy as np

def mackenzie_svp(T, S, D):
    """
    Mackenzie (1981) Sound Velocity Profile formula
    T: Temperature (C), S: Salinity (PSU), D: Depth (m)
    Returns: Sound velocity (m/s)
    """
    c = (1448.96
         + 4.591 * T
         - 5.304e-2 * T**2
         + 2.374e-4 * T**3
         + 1.340 * (S - 35)
         + 1.630e-2 * D
         + 1.675e-7 * D**2
         - 1.025e-2 * T * (S - 35)
         - 7.139e-13 * T * D**3)
    return c

def compute_svp_profile(temp_profile, sal_profile, depths):
    """
    Compute full SVP from temperature and salinity profiles.
    Returns: array of sound velocity at each depth (m/s)
    """
    return np.array([
        mackenzie_svp(T, S, D)
        for T, S, D in zip(temp_profile, sal_profile, depths)
    ])

if __name__ == '__main__':
    depths = [0, 5, 10, 20, 30, 50, 75, 100, 125, 150, 200, 300, 500, 700, 1000]
    temps  = [28, 27, 26, 25, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4]
    sals   = [35] * 15
    svp = compute_svp_profile(temps, sals, depths)
    print('SVP Profile (m/s):')
    for d, v in zip(depths, svp):
        print(f'  {d:4d}m: {v:.2f} m/s')
