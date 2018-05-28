# CAFFE for YOLO9000

## Reference

> YOLO9000: Better, Faster, Stronger

> http://pjreddie.com/yolo9000/

> https://github.com/yeahkun/caffe-yolo
## Usage

### caffe
```Shell
   vim Makefile.config
   make
```

### Data preparation
```Shell
  cd data/yolo
  mkdir lmdb
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
### Test a image
```
   cd examples/yolo/eval_detection
   jupyter notebook
   test_det.ipynb
```

### Eval VOCtest2007(The first way online)
```Shell
  # mAP reach ~56. Because of I train net poorly...you can try.
  cd examples/yolo/darknet_v3
  ./test_darknet_v3.sh
```
