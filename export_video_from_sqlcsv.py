#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Purpose: using influxDB csv results draw cow faces bbox and export to mp4

import pandas as pd
import os
import csv
import cv2
import pathlib
from tqdm import tqdm

from csv_to_hdf5 import csv_preprocess

fence_cfg = './node02_fence.csv'
bbox_cfg = './node02_bbox.csv'
img_dir = './2021_04_21/'

df = csv_preprocess(bbox_cfg)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #  iou = interArea / float(boxAArea + boxBArea - interArea)
    iou = interArea / boxBArea
    return iou

with open(fence_cfg, newline='') as f:
    rows = csv.reader(f)
    fence = [[int(x) for x in r] for r in rows]

os.chdir(img_dir)
video = []
for i, row in tqdm(df.iterrows()):
    img = cv2.imread(row[0])
    height, width, layers = img.shape
    size = (width,height)
    for f in range(len(fence)):
        cv2.rectangle(img,
                      tuple(fence[f][:2]),
                      tuple(fence[f][2:]),
                      (255, 0, 0), 2)
    for n in range(1, 5):
        cv2.rectangle(img,
                      tuple(row[n][:2]),
                      tuple(row[n][2:]),
                      (0, 255, 0), 2)
    video.append(img)

out = cv2.VideoWriter('../project.mp4', cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 3, size)

for i in range(len(video)):
    out.write(video[i])
out.release()
