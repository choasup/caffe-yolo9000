# CAFFE for YOLO

## Reference

> YOLO9000: Better, Faster, Stronger

> http://pjreddie.com/yolo9000/

> https://github.com/yeahkun/caffe-yolo
## Usage


### Data preparation
```Shell
  cd data/yolo
  ln -s /your/path/to/VOCdevkit/ .
  python ./get_list.py
  # change related path in script convert.sh
  ./convert.sh 
```

### Train
```Shell
  cd examples/yolo/darknet_v3
  # change related path in script train.sh
  mkdir models
  ./train_darknet_v3.sh
```

### Test (The first way online)
```Shell
  # if everything goes well, mAP reach ~56.
  cd examples/yolo/darknet_v3
  ./test_darknet_v3.sh
```

### Test (The second way offline)
```Shell
  cd examples/eval_detection
  python test_yolo_v2.py
```
