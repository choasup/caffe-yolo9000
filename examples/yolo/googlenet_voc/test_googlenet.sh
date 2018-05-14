#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./gnet_region_test_googlenet.prototxt
MODEL=./models/gnet_yolo_region_googlenet_iter_.caffemodel
GPU_ID=1
ITER=500

num=40
for i in `seq $num`;
do

MODEL="./models/gnet_yolo_region_googlenet_iter_"
MODEL=$MODEL$(expr $i \* 500)
MODEL=$MODEL".caffemodel"

#MODEL="./models/gnet_yolo_region_googlenet_iter_10000.caffemodel"

echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=20 --side=13
done
