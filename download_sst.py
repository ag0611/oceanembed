from datetime import date, timedelta
import urllib.request, os

save_dir = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\raw\sst'
base_url = 'https://www.ncei.noaa.gov/data/sea-surface-temperature-optimum-interpolation/v2.1/access/avhrr'

start = date(2022, 1, 2)
end   = date(2022, 12, 31)

d = start
while d <= end:
    ym = d.strftime('%Y%m')
    fname = 'oisst-avhrr-v02r01.' + d.strftime('%Y%m%d') + '.nc'
    url = base_url + '/' + ym + '/' + fname
    out = os.path.join(save_dir, fname)
    if not os.path.exists(out):
        print('Downloading ' + fname + '...')
        try:
            urllib.request.urlretrieve(url, out)
        except Exception as e:
            print('FAILED: ' + str(e))
    d += timedelta(days=1)

print('Done!')
