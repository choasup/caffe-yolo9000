#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./gnet_region_test_darknet448_voc.prototxt
MODEL=./models/gnet_yolo_region_googlenet_iter_.caffemodel
GPU_ID=2
ITER=500

num=1
for i in `seq $num`;
do

MODEL="./models/gnet_yolo_region_darknet448_voc__iter_"
MODEL=$MODEL$(expr $i \* 2500)
MODEL=$MODEL".caffemodel"

MODEL="./models/gnet_yolo_region_darknet448_voc__iter_80000.caffemodel"

echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=20 --side=17
done
