import os
import urllib.request
from datetime import date, timedelta

RAW = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\raw'

def fetch_sst(target_date):
    ym = target_date.strftime('%Y%m')
    fname = 'oisst-avhrr-v02r01.' + target_date.strftime('%Y%m%d') + '.nc'
    url = f'https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr/{ym}/{fname}'
    out = os.path.join(RAW, 'sst', fname)
    if not os.path.exists(out):
        print(f'Fetching SST {fname}...')
        urllib.request.urlretrieve(url, out)
    return out

def fetch_all(target_date=None):
    if target_date is None:
        target_date = date.today() - timedelta(days=1)
    print(f'Fetching data for {target_date}...')
    fetch_sst(target_date)
    print('Fetch complete!')

if __name__ == '__main__':
    fetch_all()
