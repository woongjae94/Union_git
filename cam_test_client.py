# Import from pkg lib
import os, sys
import cv2 as cv
import numpy as np
import argparse
import time
import pickle
import dlib  # 얼굴인식을 위한 라이브러리
from PIL import Image
import time
import requests
import socket

# Import torch base
import torch
import torch.nn as nn
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

#import TF base
import tensorflow as tf

#import from pre code
#from model.darknet.python.darknet import *
#from utils.crop_frames import CropFrames
#from model.TFModel.TFModel_socket import I3DClassifier

# Import from my lib
#from utils.cam_server_util import get_args_and_set_stream, main_discription
from utils.log_util import add_log
from model.hopenet import hopenet, hopenet_utils, hopenetlite_v2
from utils.cam_client_util import *
from utils.ip_util import get_ip_address


# Arguments
parser = argparse.ArgumentParser()

# IP address import
my_ip = get_ip_address()
#my_ip = '172.17.0.2'
parser.add_argument('--ip', default=my_ip, type=str)
parser.add_argument('--port', default='8090', type=str, help="port number for communicate")
parser.add_argument('--headpose_mode', default='lite', type=str, help="lite | normal")
parser.add_argument('--face_model', default='./model/hopenet/mmod_human_face_detector.dat', type=str, help='Path of DLIB face detection model.')
parser.add_argument('--action_crop', default=True, type=bool, help='use crop img in action model')



def set_backend_and_model(arg_mode):
    if args.headpose_mode == 'normal':
        # ResNet50 structure
        snapshot_path = './model/hopenet/hopenet_robust_alpha1.pkl'
        return snapshot_path, hopenet.Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
    elif args.headpose_mode == 'lite':
        snapshot_path = './model/hopenet/output_lite_epoch_120.pkl'
        return snapshot_path, hopenetlite_v2.shufflenet_v2_x1_0() # v3+lite = 0.25 / tiny+lite = 0.1
    else:
        print("arg_mode err, "+arg_mode)
        add_log("client", "Unexpected param in args.headpose_mode --> "+arg_mode)
        return None
    

if __name__ == "__main__":
    cudnn.enabled = True
    print(dlib.DLIB_USE_CUDA)
    
    add_log("client", "start")
    args = parser.parse_args()
    cam_address = 'http://' + args.ip + ':' + args.port + '/?action=stream'

    print("Load head pose model...")
    snapshot_path, model = set_backend_and_model(args.headpose_mode)
    saved_state_dict = torch.load(snapshot_path, map_location='cuda')
    model.load_state_dict(saved_state_dict)
    model = model.cuda()
    model.eval()

    print("Load face recognition dlib model...")
    cnn_face_detector = dlib.cnn_face_detection_model_v1(args.face_model)


    print("Load part detection model...")
    #yolo = load_net("./darknet/cfg/yolov3.cfg", "./darknet/cfg/yolov3.weights", 0)
    #meta = load_meta("./darknet/cfg/coco.data")

    print("Load Action model...")
    #if args.action_crop:
    #    action_model = I3DClassifier(os.path.join('model/I3D', 'i3d_crop'))
    #else:
    #    action_model = I3DClassifier()
    

    # data transform
    transformations = transforms.Compose([
        transforms.Resize(224), # 짧은 축의 길이를 224로 비율에 맞춰 조절
        transforms.CenterCrop(224), # 224x224 square crop
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    idx_tensor = [idx for idx in range(66)]
    idx_tensor = torch.FloatTensor(idx_tensor).cuda()

    print("Connect cam streaming server...")
    cap = cv.VideoCapture(cam_address) #mjpg-streamer 프로그램을 통해 웹에서 데이터 획득
    #cap = cv.VideoCapture(0)          #로컬 카메라 사용

    clips = []

    while(True):
        start = time.time()
        ret, frame = cap.read()

        if not ret:
            add_log("client", "connect Fail")
            print("Failed to connecting server")
            break
        #print("input frame size : ", str(frame.shape))
        new_frame = cv.resize(frame, (int(frame.shape[1]*0.5), int(frame.shape[0]*0.5)), interpolation=cv.INTER_AREA)        
        new_frame = cv.cvtColor(new_frame, cv.COLOR_BGR2RGB)

        face_detects = cnn_face_detector(new_frame, 1)
        after_face_detect = time.time()

        #clips.append(new_frame)

        for idx, det in enumerate(face_detects):
            conf = det.confidence
            #print(idx, " th confidence : ", conf)

            if conf > 0.8:
                x_min = det.rect.left()
                y_min = det.rect.top()
                x_max = det.rect.right()
                y_max = det.rect.bottom()

                
                bbox_width = abs(x_max - x_min)
                bbox_height = abs(y_max - y_min)
                x_min -= 2 * bbox_width / 4
                x_max += 2 * bbox_width / 4
                y_min -= 3 * bbox_height / 4
                y_max += bbox_height / 4
                x_min = int(max(x_min, 0))
                y_min = int(max(y_min, 0))
                x_max = int(min(new_frame.shape[1], x_max))
                y_max = int(min(new_frame.shape[0], y_max))

                # Crop image
                img = new_frame[y_min:y_max,x_min:x_max]
                img = Image.fromarray(img)

                # Transform
                img = transformations(img)
                img_shape = img.size()
                img = img.view(1, img_shape[0], img_shape[1], img_shape[2])
                img = Variable(img).cuda()

                yaw, pitch, roll = model(img)
                after_hopenet = time.time()

                yaw_predicted = F.softmax(yaw)
                pitch_predicted = F.softmax(pitch)
                roll_predicted = F.softmax(roll)
                
                # Get continuous predictions in degrees.
                yaw_predicted = torch.sum(yaw_predicted.data[0] * idx_tensor) * 3 - 99
                pitch_predicted = torch.sum(pitch_predicted.data[0] * idx_tensor) * 3 - 99
                roll_predicted = torch.sum(roll_predicted.data[0] * idx_tensor) * 3 - 99

                hopenet_utils.draw_axis(new_frame, yaw_predicted, pitch_predicted, roll_predicted, tdx = (x_min + x_max) / 2, tdy= (y_min + y_max) / 2, size = bbox_height/2)
                after_hopenet_draw = time.time()
                cv.rectangle(new_frame, (x_min, y_min), (x_max, y_max), (0,255,0), 1)
                after_rec_draw = time.time()
                """
                print("Duration -  ~ face detect : ", after_face_detect - start)
                print("         -  ~ hope net    : ", after_hopenet - after_face_detect)
                print("         -  ~ net draw    : ", after_hopenet_draw - after_hopenet)
                print("         -  ~ rec draw    : ", after_rec_draw - after_hopenet_draw)
                
                print("yaw_predicted: " + str(yaw_predicted))
                print("pitch_predicted: " + str(pitch_predicted))
                print("roll_predicted: " + str(roll_predicted))
                """
                
                if(yaw_predicted < -50 ):
                    print("왼쪽 멀리")
                elif(yaw_predicted <= -20):
                    print("왼쪽 ")
                elif(yaw_predicted <= 10):
                    print("중앙")
                elif(yaw_predicted<=35):
                    print("오른쪽")
                elif(yaw_predicted>45):
                    print("오른쪽 멀리")

        frame = cv.cvtColor(new_frame, cv.COLOR_RGB2BGR)
        frame = cv.resize(frame, (int(frame.shape[1]*2), int(frame.shape[0]*2)), interpolation=cv.INTER_AREA)
        cv.imshow("test", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()
