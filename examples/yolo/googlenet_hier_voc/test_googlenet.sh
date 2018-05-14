#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./gnet_region_test_googlenet.prototxt
MODEL=./models/gnet_yolo_region_googlenet_iter_75000.caffemodel
GPU_ID=3
ITER=100

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID 2>&1 | tee test.log

