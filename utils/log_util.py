import os
import time

today = time.strftime('%Y-%m-%d', time.localtime(time.time()))
time_now = time.strftime('%H:%M:%S', time.localtime(time.time()))

def add_log(source, msg):
    with open('./log/'+today+".txt", "a") as f:
        f.write(today+'\t'+time_now+'\t'+source+'\t'+msg+'\n')