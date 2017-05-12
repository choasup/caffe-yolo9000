# I am sorry that I have to close this repository for some reasons. If you are interested in YOLO9000, you can contact with me. 

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
#### MODEL=./models/gnet_yolo_region_darknet_v3_pretrain_rectify_iter_200000.caffemodel
#### model is here:
> https://pan.baidu.com/s/1nvHggFB     t7ui

### Eval VOCtest2007(The second way offline)
```Shell
  cd examples/eval_detection
  python test_yolo_v2.py
```

### Draw loss figure(avg_obj, avg_noobj, avg_class, avg_iou, recall)
```
  cd tools/yolo_extra
  python parse_log_yolo.py ./log/train_darknet_anchor.log ./log
```

#### If you want to train your datasets. you should edit the train_prototxt. 
```
   [conv_reg layer] num_output = num * (num_class + coords + 1) = 5 * (your_classes_num + 4 + 1)
   [det_loss layer] num_class = your_classes_num
```

##### yolo9000-Tree example prototxts, model and .sh will update soon!
