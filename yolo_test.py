import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import torch
from ultralytics.data.augment import DctElaTransform, Format

# # Load the model.
# model = YOLO('yolo11m.pt') # pretrained model 

# # Training.
# results = model.train(
#     data='data/yolo/dataset.yaml',  # .yaml file 
#     imgsz=640, # image size
#     epochs=100,  # epoch number
#     batch=48, # batch size , I normally use 8 or 16 but my GPU gave memory errors, therefore I reduced it to 4 for this time.
#     name='tamper_11m_50epochs', # output folder name, it contains model weights and all of the other things.
#     plots=True, # Plots about metrics (precision, recall,f1 score)
#     amp=True, # amp=True gives me an error, I don't know why , If it doesn't give you an error, set it to True
#     close_mosaic=25,
#     single_cls=True,
# )

task_name = 'yolo11m_2b_biformer'
# model = YOLO('yolo11n-2branch.yaml').to('cuda')
# model.load(os.path.join('runs/detect', task_name, 'weights', 'best.pt'))
# model = YOLO(os.path.join('runs/detect', task_name, 'weights', 'best.pt'))
model = YOLO(os.path.join('runs/detect/extra', task_name, 'weights', 'best.pt'))
# Output Detection for Test Images
test_dir = "data/val_images"

js = {}
threshold = 0.5

for pic in tqdm(os.listdir(test_dir)):
    img = cv2.imread(os.path.join(test_dir, pic))

    # result = model.predict(img)[0]
    result = model.predict(img,max_det=1,
        conf=threshold,device='cuda',
        verbose=False, 
        )[0]
    region = result.boxes.xyxy
    bbox = np.array(region.cpu()).tolist()
    js[pic] = bbox

import json

with open(f"output/result_{task_name}.json", "w") as f:
    f.write('[')
    for pic in js:
        f.write(json.dumps({"id": pic, "region": js[pic]}) + ",\n")
    # Remove the last comma
    f.seek(f.tell() - 2)
    f.write(']')