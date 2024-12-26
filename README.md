# Document Tampering Detection YOLO

This is the implementation of the final PJ for Fudan Computer Vision lecture.

Authors: 

## Training

Modified training args in yolo_train.py and run

```
python yolo_train.py
```

_To avoid extra transforms and inputs (e.g. ELA, BlockDCT) , please modify codes in_

_ultralytics/data/dataset.py: line 65_

```python
self.extra = False
```

## Testing

Modified testing args in yolo_test.py and run

```
python yolo_test.py
```

You will get a json file recording the detected bboxes of images in test directory.

_To avoid extra transforms and inputs (e.g. ELA, BlockDCT) , please modify codes in_

_ultralytics/engine/predictor.py: line 115_

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

