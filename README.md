# CAFFE for YOLO9000

## Reference

> YOLO9000: Better, Faster, Stronger

> http://pjreddie.com/yolo9000/

> https://github.com/yeahkun/caffe-yolo

> https://github.com/weiliu89/caffe/tree/ssd
## Usage

### Caffe 
```Shell
   vim Makefile.config
   make clean
   make all -j8
   make pycaffe
```
(Be careful: caffe&caffe2 PYTHONPATH conflict)

### Data preparation
Like SSD data setting.
```Shell
  cd data/VOC0712
  
  vim create_data.sh
  ./create_data.sh 
```

### Train
```Shell
  cd examples/yolo/darknet_voc
  mkdir models
  
  # edit according yourself
  vim gnet_region_train_darknet448_voc.prototxt
  vim gnet_region_test_darkent448_voc.prototxt
  vim gnet_region_solver_darkent448_voc.prototxt

  # download pretrain_model
```  
  > https://pan.baidu.com/s/1c71EB-6A1xQb2ImOISZiHA password: 9u5v
```
  # change related path in script train.sh
  vim train_darknet448.sh
  
  ./train_darknet448.sh
```

### Test a image
```Shell
   vim test_darknet448.sh
   ./test_darknet448.sh
```

### Eval mAP
```Shell
   cd demo
   vim demo_yolo.py
   python demo_yolo.py
```
