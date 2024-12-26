import os
import json
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from ultralytics import YOLO
import cv2
import torch
from PIL import Image, ImageDraw
from ultralytics.data.augment import DctElaTransform, Format


def draw_bbox(img_path, pred, labels, out_dir, conf=None, pred_color=(255, 0, 0), label_color=(0, 255, 0), thickness=2):
    """
    绘制预测框和标签框，并将结果保存到out_dir目录中。
    
    Args:
        img_path (str): 输入图像路径。
        pred (list): 预测框列表，每个元素为(x1, y1, x2, y2)。
        labels (list): 标签框列表，每个元素为(x1, y1, x2, y2)。
        out_dir (str): 输出目录路径，保存绘制后的图像。
        pred_color (tuple): 预测框的颜色，默认为红色 (255, 0, 0)。
        label_color (tuple): 标签框的颜色，默认为绿色 (0, 255, 0)。
        thickness (int): 框线的厚度，默认为 2。
    """
    # 创建输出目录，如果不存在则创建
    os.makedirs(out_dir, exist_ok=True)
    
    # 读取输入图像
    img = Image.open(img_path)
    draw = ImageDraw.Draw(img)
    
    # 绘制预测框 (pred) - 红色
    for it, bbox in enumerate(pred):
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=pred_color, width=thickness)
        conf_bbox = conf[it]
        draw.text((x1, y1), f'Conf:{conf_bbox:.2f}', fill='red')
    
    # 绘制标签框 (labels) - 绿色
    for bbox in labels:
        if bbox is None:
            continue
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=label_color, width=thickness)
    
    # 生成输出图像文件路径
    img_name = os.path.basename(img_path)
    out_path = os.path.join(out_dir, img_name)
    
    # 保存绘制后的图像
    img.save(out_path)

def xywh2xyxy(box, size):
    x, y, w, h = box
    x1 = (x - w / 2) * size[0]
    y1 = (y - h / 2) * size[1]
    x2 = (x + w / 2) * size[0]
    y2 = (y + h / 2) * size[1]
    return [x1, y1, x2, y2]

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

# task_name = '11n_2b_down2'
task_name = '11n_2b_7c'

outdir = f"output/{task_name}"

# model = YOLO('yolo11n-2branch.yaml').to('cuda')
# model.load(os.path.join('runs/detect', task_name, 'weights', 'best.pt'))
model = YOLO(os.path.join('runs/detect', task_name, 'weights', 'best.pt'))

# Output Detection for Test Images
test_dir = "data/yolo/images/val"
import random
vis_list = random.sample(os.listdir(test_dir), 50)
with open("data/vis_images.txt", "w") as f:
    for pic in vis_list:
        f.write(pic + "\n")

js = {}
threshold = 0.2

for pic in tqdm(vis_list):
    img = cv2.imread(os.path.join(test_dir, pic))
    lable_file = os.path.join(test_dir.replace('images', 'labels'), pic.replace('.jpg', '.txt'))
    with open(lable_file, 'r') as f:
        lines = f.readline().strip()
    if len(lines)>0:
        label = [float(x) for x in lines.split(' ')[1:]]
        label = xywh2xyxy(label, (img.shape[1], img.shape[0]))
    else:
        label = None
    # result = model.predict(img)[0]
    result = model.predict(img,max_det=1,
        conf=threshold,device='cuda',
        verbose=False, 
        # save=True, 
        # name=f'output',
        # save_conf=True,
        )[0]
    region = result.boxes.xyxy
    conf = result.boxes.conf.cpu().tolist()
    bbox = np.array(region.cpu()).tolist()
    js[pic] = bbox
    draw_bbox(os.path.join(test_dir, pic), bbox, [label], outdir, conf)

# with open("data/result_11n_2b_down.json", "w") as f:
#     f.write('[')
#     for pic in js:
#         f.write(json.dumps({"id": pic, "region": js[pic]}) + ",\n")
#     # Remove the last comma
#     f.seek(f.tell() - 2)
#     f.write(']')

