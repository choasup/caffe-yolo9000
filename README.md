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
  # if everything goes well, mAP reach ~56.
  cd examples/yolo/darknet_v3
  ./test_darknet_v3.sh
```
#### MODEL=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_200000.caffemodel
#### model is here:
> https://pan.baidu.com/s/1nvHggFB     t7ui

### Eval VOCtest2007(The second way offline)
```Shell
  cd examples/eval_detection
  python test_yolo_v2.py
```

##### yolo9000-Tree example prototxts, model and .sh will update soon!
