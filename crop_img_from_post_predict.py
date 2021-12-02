#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Purpose: using GPU detect batch node's images and Crop cow faces and put into
#   corresponding fence folder.

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
import cv2
import csv
import pathlib
from datetime import datetime
from tqdm import tqdm
import numpy as np

from utils import parse_params


def imgPre_c(image):
    """
    img preprocessing to feed into yolo
    """
    image = image.astype(np.float32)
    image = image[:,:,::-1]
    image = cv2.resize(((image/255.0)-0.5)*2.0, (224, 224))
    return np.expand_dims(image, axis=0)

def iou(boxA, boxB):
    """
    calculate cow bbox and fence iou area
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / boxBArea
    return iou

if __name__ == "__main__":

################################################################################
#                             global config params                             #
################################################################################
    node = '01'
    m = '02'
    d = '01'
    fence_cfg = f'./cfg/{m}{d}-node{node}_fence.csv'

    start_time = datetime(1900, 1, 1, 11, 33, 16)
    end_time = datetime(1900, 1, 1, 12, 30, 6)

    img_dir = f'./IMG/NODE{node}/2021_{m}_{d}/'
    ref_dict_path = './cfg/ref_dict.csv'
    ref_vec_path = './cfg/ref.8f.tsv'

    # read params path
    params_path = './weight/'
    params = parse_params(params_path)

################################################################################
#                               build yolo model                               #
################################################################################
    weight = "./weight/yolov4-tiny-CowFace-anch-default_best.weights"
    cfg = "./weight/yolov4-tiny-CowFace-anch-default.cfg"
    net = cv2.dnn.readNet(weight, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=(416, 416), scale=1./255, swapRB=True)

################################################################################
#                              load fence config                               #
################################################################################
    with open(fence_cfg, newline='') as f:
        rows = csv.reader(f)
        fence = [[int(x) for x in r] for r in rows]

################################################################################
#                          load cow database feature                           #
################################################################################
    refs = np.loadtxt(ref_vec_path, dtype=np.float16, delimiter='\t')

################################################################################
#                        select img between valid time                         #
################################################################################
    os.chdir(img_dir)
    files = sorted(os.listdir())
    sel_files = []
    for name in tqdm(files):
        file_time = datetime.strptime(name, f"2021_{m}_{d}-%H_%M_%S.jpg")
        if start_time < file_time and file_time < end_time:
            sel_files.append(name)

################################################################################
#                 create folder corresponding to fence number                  #
################################################################################
    [pathlib.Path(f"{i:02d}").mkdir(exist_ok=True) for i in range(len(fence))]

################################################################################
#                     strat predicting and saving crop img                     #
################################################################################
    #  for file in tqdm(glob.glob("*.jpg")):
    for i, file in tqdm(enumerate(sel_files)):
        frame = cv2.imread(file)

        # Predict bbox
        classes, scores, boxes = model.detect(frame, 0.6, 0.4)

        for (classid, score, box) in zip(classes, scores, boxes):
            (x,y,w,h) = box

            ious = [iou(f,(x,y,x+w,y+h)) for f in fence]
            if(max(ious) < 0.5):
                continue
            which_f = ious.index(max(ious))
            cv2.imwrite(f'{which_f:02d}/{node}_{m}{d}{i:03d}.jpg',
                        frame[y:y+h, x:x+w])

