# Document Tampering Detection YOLO

This is the implementation of the final PJ for Fudan Computer Vision lecture.

Authors: 

## Training

Modified training args in yolo_train.py and run

```
python yolo_train.py
```

_To avoid extra transforms and inputs (e.g. ELA, BlockDCT) , please modify codes in_

1. _ultralytics/data/dataset.py: line 185_

```python
# transforms.append(DctElaTransform())
```

2. _ultralytics/data/dataset.py: line 196_

```python
bgr=hyp.bgr if self.augment else 0.0,  # only affect training.
# bgr=1.0
```

## Testing

Modified testing args in yolo_test.py and run

```
python yolo_test.py
```

You will get a json file recording the detected bboxes of images in test directory.

_To avoid extra transforms and inputs (e.g. ELA, BlockDCT) , please modify codes in_

1. _ultralytics/engine/predictor.py: line 115_

```python
self.extra = False
```

## Visualization

Visualize the detection results on validation set for debugging

```
python vis.py
```

## Model 

Our model configs saved in directory 'config' are in YOLO yaml format.

For custom modules you may reference _ultralytics/nn/extra.py_

