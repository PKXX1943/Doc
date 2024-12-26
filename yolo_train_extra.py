import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import torch

# Load the model.
scale = 'm'
base_model = f'yolo11{scale}'
model_name = f'config/{base_model}_2b'
model = YOLO(f'{model_name}.yaml')
# ckpt_idx help custom model to load the pretrained YOLO weights from original layer indexes.
model.load(f'ckpt/{base_model}.pt', ckpt_idx=[6,7,8,10,11] + [i for i in range(13, 32)])

# Training.
results = model.train(
    data='data/yolo/dataset.yaml',  # .yaml file 
    imgsz=640, # image size
    epochs=100,  # epoch number
    batch=8, # batch size , I normally use 8 or 16 but my GPU gave memory errors, therefore I reduced it to 4 for this time.
    name=model_name.split('/')[-1], # output folder name, it contains model weights and all of the other things.
    plots=True, # Plots about metrics (precision, recall,f1 score)
    amp=False, # amp=True gives me an error, I don't know why , If it doesn't give you an error, set it to True
    close_mosaic=50,
    single_cls=True,
    cls=0.1,
    box=10,
)
