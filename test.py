import os
import argparse
import socket
import time
import subprocess
import webbrowser
import sys
import time

# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--fps', default='20', type=str, help="Frame per Second, maximum 30")
parser.add_argument('--res', default='qVGA', type=str, help="resolution of cam / details in \"help\" at main")
parser.add_argument('--port', default='5000', type=str, help="port number for communicate")
parser.add_argument('--multi', default=False, type=bool, help="If you use multi cam, Set True")

#with open("./cam/cam_streaming.sh", 'w') as f:
#    f.write("mjpg_streamer -i \"input_uvc.so -f 10 -r 320x240\" -o \"output_http.so -p 5000 -w /usr/local/share/mjpg-streamer/www/\"")

args = parser.parse_args()

# args 사전형 확인
a=args.__dict__
print(list(a.items()))

for item in a.items():
    print(item[0], " : ", item[1])

print(sys.version)

# 내 ip 주소 얻어오기
# 이더넷 고정 ip 아래처럼 얻을 수 있음, 출력 ip는 str 타입
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
s.connect(('8.8.8.8', 0))
ip = s.getsockname()[0]
print(ip, "ok")

print(time.strftime('%Y-%m-%d %H:%M:%S'))
print(time.strftime('%Y-%m-%d', time.localtime(time.time())))
print(time.strftime('%H:%M:%S', time.localtime(time.time())))

import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Worker : ", device)
os.system("nvcc --version")

#import tensorflow
#print("tensorflow ok")

import numpy as np
print("np ok")

from flask import Flask, render_template, request, Response, jsonify
import logging
print("flask ok")

import requests
print("requests ok")


import cv2
print(cv2.__version__)

import dlib
print("dlib ok")


#import Jetson.GPIO as GPIO
#print("set")
mills = lambda: int(round(time.time() * 1000))
prev = mills()
cap = cv2.VideoCapture('http://127.0.0.1:8090/?action=stream')
while cap.isOpened():
    now = mills()
    ret, frame = cap.read()
    print(prev, now, now - prev)
    if now - prev > 100:
        prev = now
        if ret:
            cv2.imshow("window",frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
cap.release()
cv2.destroyAllWindows()

