#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./yolov2-544_test.prototxt
GPU_ID=3
ITER=500

num=1
for i in `seq $num`;
do

MODEL="./yolov2-544-voc/yolov2-544-voc.caffemodel"
echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=20 --side=13
done
