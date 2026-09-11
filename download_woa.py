import urllib.request
import ssl
import os

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/temperature/netcdf/decav/0.25/woa23_decav_t00_04.nc"
out = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\climatology\woa23_NIO.nc'

print("Downloading WOA23...")
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
with opener.open(url) as r:
    with open(out, 'wb') as f:
        f.write(r.read())
print("Done!", os.path.getsize(out) // (1024*1024), "MB")
