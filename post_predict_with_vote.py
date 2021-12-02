#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  #supress tensorflow info except error
os.environ["CUDA_VISIBLE_DEVICES"]= "1"
import cv2
import csv
import sys
import glob
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import tensorflow as tf
from tqdm import tqdm
from model.parse_params import parse_params
from model.triplet_model_fn import model_fn


def imgPre_c(image):
    image = image.astype(np.float32)
    image = image[:,:,::-1]
    image = cv2.resize(((image/255.0)-0.5)*2.0, (224, 224))
    return np.expand_dims(image, axis=0)

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / boxBArea
    return iou

if __name__ == "__main__":
    node = f'{int(sys.argv[1]):02d}'
    print(node)
    m = '03'
    d = '19'
    fence_cfg = f'./cfg/node{node}_fence.csv'
    threshold = 0.676
    size = (416, 416)
    img_dir = f'/home/ubuntu/Analyze_data/IMG/NODE{node}/2021_{m}_{d}/'
    out_dir = f'/home/ubuntu/Analyze_data/OUT_IMG/NODE{node}/2021_{m}_{d}/'
    ref_dict_path = './cfg/ref_dict.csv'
    ref_vec_path = './cfg/ref.8f.tsv'
    mobilenet_param_path = './weight/'
    weight = "./weight/yolov4-tiny-CowFace-anch-default_best.weights"
    cfg = "./weight/yolov4-tiny-CowFace-anch-default.cfg"

    start_time = datetime(1900, 1, 1, 6, 0, 0)
    end_time = datetime(1900, 1, 1, 18, 0, 0)


    # outdir
    outdir = pathlib.Path(out_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # build mobilenet model
    params = parse_params(mobilenet_param_path)
    m_model = model_fn(params, is_training=False)
    m_model.load_weights(os.path.join(mobilenet_param_path, 'model'))

    # build yolo model
    net = cv2.dnn.readNet(weight, cfg)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
    model = cv2.dnn_DetectionModel(net)
    model.setInputParams(size=size, scale=1./255, swapRB=True)

    # fence_cfg
    with open(fence_cfg, newline='') as f:
        rows = csv.reader(f)
        fence = [[int(x) for x in r] for r in rows]

    # ref_dict
    ref_dict = dict()
    with open(ref_dict_path, newline='') as f:
        rows = csv.reader(f)
        for i, r in enumerate(rows):
            ref_dict[int(i)] = r[0]

    # Open reference feature
    refs = np.loadtxt(ref_vec_path, dtype=np.float16, delimiter='\t')


#=================== Enter img dir, start predict ============================#
    os.chdir(img_dir)

    files = sorted(os.listdir())
    sel_files = []
    print("select valid file between 06am ~ 18pm")
    pre_time = datetime.strptime(files[0], f"2021_{m}_{d}-%H_%M_%S.jpg")
    sel_files.append(pre_time.strftime(f"2021_{m}_{d}-%H_%M_%S"))
    for name in tqdm(files[1:]):
        file_time = datetime.strptime(name, f"2021_{m}_{d}-%H_%M_%S.jpg")
        if start_time < file_time and file_time < end_time:
            if (file_time - pre_time) > timedelta(0,7):
                sel_files.append((file_time - timedelta(0,1)).strftime(f"2021_{m}_{d}-%H_%M_%S"))
                sel_files.append((file_time - timedelta(0,2)).strftime(f"2021_{m}_{d}-%H_%M_%S"))
            sel_files.append(name)
        pre_time = file_time

    column = (
        [f"f{x}" for x in range(len(fence))] +
        [f"f{x}-id" for x in range(len(fence))] +
        list(ref_dict.values())
    )
    preAllocate = np.zeros((len(sel_files), len(column)), dtype=int)
    df = pd.DataFrame(preAllocate, columns=column, index=sel_files)
    preAllocate = np.zeros((len(sel_files), len(fence)), dtype=float)
    df_dist = pd.DataFrame(preAllocate, columns=[x for x in range(len(fence))], index=sel_files)

    column = (
        [f"f{x}-box" for x in range(len(fence))] +
        [f"f{x}-feat" for x in range(len(fence))]
    )
    preAllocate = np.zeros((len(sel_files), len(column)), dtype=float)
    df_feat = pd.DataFrame(preAllocate, columns=column, index=sel_files).astype(object)

    for file in tqdm(sel_files):
        if '.jpg' not in file:
            continue
        image = cv2.imread(file)
        frame = np.copy(image)

        # Predict bbox
        classes, scores, boxes = model.detect(frame, 0.6, 0.4)
        for (classid, score, box) in zip(classes, scores, boxes):
            (x,y,w,h) = box

            ious = [iou(f,(x,y,x+w,y+h)) for f in fence]
            if(max(ious) < 0.5):
                continue
            which_f = ious.index(max(ious))
            df[f"f{which_f}"][file] = 1
            df_feat[f"f{which_f}-box"][file] = box

            # Export feature
            c_out = m_model.predict(imgPre_c(frame[y:y+h, x:x+w]))[0]
            df_feat[f"f{which_f}-feat"][file] = c_out

            vec = np.expand_dims(c_out, axis=0)
            dists = (-2*np.dot(vec, refs.T)
                  + np.sum(refs**2, axis=1)
                  + np.sum(vec**2, axis=1)[:,np.newaxis])
            min_index = np.argmin(dists, axis=1)[0]
            ID = ref_dict[min_index]

            df_dist[which_f][file] = np.amin(dists)
            if np.amin(dists) < threshold:
                df[ID][file] = 1
                df[f"f{which_f}-id"][file] = int(ID)
            else:
                df[f"f{which_f}-id"][file] = -3


    df.to_csv(outdir.parent.absolute().joinpath(f"full.{node}-2021_{m}_{d}.csv"))
    df_dist.to_csv(outdir.parent.absolute().joinpath(f"dist.{node}-2021_{m}_{d}.csv"))
    df_feat.to_pickle(outdir.parent.absolute().joinpath(f"feat.{node}-2021_{m}_{d}.pkl"))

#=================== Voting system ===========================================#
    df = pd.read_csv(outdir.parent.absolute().joinpath(f"full.{node}-2021_{m}_{d}.csv"), index_col=0)
    print("start voting")
    df.loc[:, '10008':'10706'] = 0
    fs = ['f0','f1','f2']
    def check(t, result):
        for l, r in result:
            if l < s and s < r:
                return False
        return True

    for f in tqdm(fs):
        pattern = [0,1,1]
        start = [
            df.index[i - len(pattern) + 1] # Get the datetime index
            for i in range(len(pattern), len(df)) # For each 3 consequent elements
            if all(df[f][i-len(pattern):i] == pattern) # If the pattern matched
        ]
        pattern = [1,0,0]
        end = [
            df.index[i - len(pattern)] # Get the datetime index
            for i in range(len(pattern), len(df)) # For each 3 consequent elements
            if all(df[f][i-len(pattern):i] == pattern) # If the pattern matched
        ]
        result = []
        for e in end:
            if e > start[0]:
                result.append((start[0], e))
                break

        for s in start[1:]:
            # check if s in result range
            if check(s, result):
                arr = [e for e in end if e > s]
                if len(arr) > 0:
                    result.append((s, arr[0]))
                elif len(arr) == 0:
                    result.append((s, df.index[-1]))

        for (s,e) in result:
            if df.loc[s:e, [f'{f}-id']].squeeze().value_counts(normalize=True).max() < 0.6:
                # if top 2 cnt has -3 -> this range is -3
                if -3 in df.loc[s:e, [f'{f}-id']].squeeze().value_counts(normalize=True).nlargest(2).index:
                    df.loc[s:e, [f'{f}-id']] = -3
            elif df.loc[s:e, [f'{f}-id']].squeeze().value_counts(normalize=True).max() >= 0.6:
                cls = df.loc[s:e, [f'{f}-id']].squeeze().value_counts(normalize=True).idxmax()
                df.loc[s:e, [f'{f}-id']] = cls

    for index, row in df.iterrows():
        for f in fs:
            if df.loc[index, f'{f}-id'] != -3 and df.loc[index, f'{f}-id'] != 0:
                df.loc[index, str(df.loc[index, f'{f}-id'])] = 1
    df.to_csv(outdir.parent.absolute().joinpath(f"vote.{node}-2021_{m}_{d}.csv"))

#  #============================ Drawing box ====================================#
    df = pd.read_csv(outdir.parent.absolute().joinpath(f"full.{node}-2021_{m}_{d}.csv"), index_col=0)
    df_box = pd.read_pickle(outdir.parent.absolute().joinpath(f"feat.{node}-2021_{m}_{d}.pkl"))
    df_vote = pd.read_csv(outdir.parent.absolute().joinpath(f"vote.{node}-2021_{m}_{d}.csv"), index_col=0)
    os.chdir(img_dir)
    fs = ['f0','f1','f2']
    for index, row in tqdm(df_box.iterrows()):
        if '.jpg' not in index:
            continue
        image = cv2.imread(index)
        for f in fence:
            cv2.rectangle(image, (f[0], f[1]), (f[2], f[3]), (0,255,0), 2)
        for f in fs:
            if isinstance(df_box.loc[index, f'{f}-box'], float):
                continue
            x, y, w, h = df_box.loc[index, f'{f}-box']
            cv2.rectangle(image, (x, y, w, h), (255,255,0), 2)
            cv2.putText(image, "Vote:"+str(df_vote.loc[index, f'{f}-id']), (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            cv2.putText(image, "Orig:"+str(df.loc[index, f'{f}-id']), (x+140, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,150,0), 2)
        cv2.imwrite(os.path.join(out_dir,index), image)
