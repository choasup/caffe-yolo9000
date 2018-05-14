#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER=./ResNet-101-solver.prototxt
WEIGHTS=./ResNet-101-model.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER \
    --weights=$WEIGHTS \
    --gpu=1 2>&1 | tee train_fixed.log

#$CAFFE_HOME/build/tools/caffe train \
#    --solver=$SOLVER2 \
#    --weights=$WEIGHTS2 \
#    --gpu=3

