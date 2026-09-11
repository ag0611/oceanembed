import urllib.request
import ssl
import os

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://erddap.ifremer.fr/erddap/tabledap/ArgoFloats.nc?cycle_number,direction,time,latitude,longitude,pres,temp,psal&latitude>=5&latitude<=30&longitude>=45&longitude<=105&time>=2022-01-01T00:00:00Z&time<=2022-12-31T23:59:59Z&pres<=1000"

out = r'C:\Users\un.31ZSI.042\Desktop\OceanEmbed\data\argo\validation_profiles.nc'
print("Downloading ARGO profiles...")

opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))
with opener.open(url) as response:
    with open(out, 'wb') as f:
        f.write(response.read())

print("Done! Size:", os.path.getsize(out) // (1024*1024), "MB")
