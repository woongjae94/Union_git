import numpy as np
import cv2
import math
import requests


def normalize_probs(x):
    normalized = np.exp(x)/np.sum(np.exp(x))
    return normalized.tolist()

def sampling_frames(input_frames, sampling_num):
    total_num = len(input_frames)

    interval = 1
    if len(input_frames) > sampling_num:
        interval = math.floor(float(total_num) / sampling_num)
    print("sampling interval : {}".format(interval))
    interval = int(interval)

    out_frames = []
    for n in range(min(len(input_frames), sampling_num)):
        out_frames.append(input_frames[n*interval])

    # padding
    if len(out_frames) < sampling_num:
        print("before padding : {}".format(len(out_frames)))
        for k in range(sampling_num - len(out_frames)):
            out_frames.append(input_frames[-1])

    return out_frames

def calc_framediff(clip):
    value_list = []
    for idx in range(len(clip)-1):
        prev_frame = clip[idx]
        prev_frame = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_frame = clip[idx+1]
        curr_frame = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

        frame_diff = cv2.absdiff(prev_frame, curr_frame)
        frame_diff = np.array(frame_diff, dtype=np.int32)

        value_list.append(np.sum(frame_diff))

    # scaling values to [0-1] range using formula
    for i in range(len(value_list)):
        try:
            value_list[i] = float(value_list[i] - np.min(value_list)) / (np.max(value_list) - np.min(value_list))
        except:
            value_list[i] = .0

    return sum(value_list)

def pred_action_crop(frames, model):
    result, confidence_3, top_3 = model.run_demo_wrapper(np.expand_dims(frames, 0))

    if confidence_3[0] > 0.8:
        print(result, confidence_3[0], top_3)

    else:
        result = 'None'

    requests.get('http://155.230.104.191:5000/state/set/action', params={'action': result})
    requests.get('http://155.230.104.191:5000/state/set/action_panel', params={'labels': str(top_3),
                                                                         'probs': str(normalize_probs(confidence_3))})

    return result, confidence_3, top_3