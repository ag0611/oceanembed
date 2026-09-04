import schedule
import time
from fetcher import fetch_all

def job():
    print('Running daily fetch job...')
    fetch_all()

schedule.every().day.at("06:00").do(job)
print('Scheduler started — runs daily at 06:00 UTC')

while True:
    schedule.run_pending()
    time.sleep(60)
