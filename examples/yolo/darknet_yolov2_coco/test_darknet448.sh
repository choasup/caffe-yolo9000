#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./yolov2-608_test.prototxt
GPU_ID=3
ITER=5000

num=1
for i in `seq $num`;
do

MODEL="./yolov2-608/yolov2-608.caffemodel"
echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=80 --side=19
done
