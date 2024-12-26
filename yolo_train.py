import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import torch

# Load the model.
# model = YOLO('yolo11n-extra.yaml') 
# model.load('yolo11n.pt', [i for i in range(2, 26)])

# model = YOLO('yolo11m.pt')
scale = 'n'
model = YOLO(f'config/yolo11{scale}-2b.yaml')
model.load(f'ckpt/yolo11{scale}.pt', ckpt_idx=[i for i in range(12, 23)] + [23,25,26,27,29,30,31,32,33,34,36,37,38])

# model=YOLO('runs/detect/11n_2b_down/weights/best.pt')
# Training.
results = model.train(
    data='data/yolo/dataset.yaml',  # .yaml file 
    imgsz=640, # image size
    epochs=80,  # epoch number
    batch=32, # batch size , I normally use 8 or 16 but my GPU gave memory errors, therefore I reduced it to 4 for this time.
    name='test', # output folder name, it contains model weights and all of the other things.
    plots=True, # Plots about metrics (precision, recall,f1 score)
    amp=False, # amp=True gives me an error, I don't know why , If it doesn't give you an error, set it to True
    close_mosaic=40,
    single_cls=True,
    cls=0.1,
    box=10,
    
    # optimizer = 'AdamW',
    # lr0 = 1e-4,
    # lrf = 0.01
    # resume = True,


)
