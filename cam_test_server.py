# Import from pkg lib
import os
import cv2 as cv
import numpy as np
import subprocess
import time
import argparse
import webbrowser
import socket

# Import from my lib
from utils.cam_server_util import get_args_and_set_stream, main_discription
from utils.log_util import add_log
from utils.ip_util import get_ip_address


# Arguments
parser = argparse.ArgumentParser()

parser.add_argument('--fps', default='30', type=str, help="Frame per Second, maximum 30")
parser.add_argument('--res', default='VGA', type=str, help="resolution of cam / details in \"res\" at main")
parser.add_argument('--port', default='8090', type=str, help="port number for communicate")
parser.add_argument('--multi', default=False, type=bool, help="If you use multi cam, Set True")


# IP address import
my_ip = get_ip_address()
#my_ip = '172.17.0.2'
parser.add_argument('--ip', default=my_ip, type=str)

# main
if __name__ == "__main__":
    add_log("cam_server", "start")
    args = parser.parse_args()
    args_and_stream = get_args_and_set_stream(args.__dict__)
    main_notice = main_discription(args.__dict__)

    print("...Cam Streaming server setting--------")
    args_and_stream.write_setting_to_file()
    main_notice.print_info()

    print("...Cam Streaming server Start ---------")
    cam_start = subprocess.Popen(["sh", "./cam/cam_streaming.sh"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, stdin=subprocess.PIPE)
    time.sleep(1)

    cam_address = 'http://' + args.ip + ':' + args.port + '/?action=stream'
    cap = cv.VideoCapture(cam_address)
    print("...Cam Streaming server Connect -------")
    
    while(True):
        main_notice.print_main()
        key_in = input("input key : ")

        if key_in == 'quit':
            os.system('killall -9 mjpg_streamer')
            print("mjpg streamer process kill")
            add_log("cam_server", "Shutdown")
            break
        elif key_in == 'info':
            main_notice.print_info()
        elif key_in == 'res':
            main_notice.print_res()
        elif key_in == 'open':
            url = 'http://' + my_ip + ':' + args.port + '/stream.html'
            webbrowser.open(url)
        elif key_in == 'ping':
            ret, frame = cap.read()
            if not ret:
                print("fail")
            else:
                print("read sucess", frame.shape)
        else:
            continue
    
    print("camera streaming shutdown")
