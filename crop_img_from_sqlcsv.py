import pandas as pd
import os
import csv
import cv2
import pathlib
from tqdm import tqdm

from utils import csv_preprocess

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
[pathlib.Path(f"{i:02d}").mkdir(exist_ok=True) for i in range(len(fence))]

for i, row in tqdm(df.iterrows()):
    img = cv2.imread(row[0])
    for n in range(1, 5):
        if row[n][0] == -1:
            continue
        ious = [iou(x, row[n]) for x in fence]
        max_id = ious.index(max(ious))
        cv2.imwrite(f'{max_id:02d}/{i:03d}.jpg',
                    img[row[n][1]:row[n][3], row[n][0]:row[n][2]])
