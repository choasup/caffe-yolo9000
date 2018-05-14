#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER1=./ResNet-101-solver.prototxt
#SOLVER2=./ResNet-101-solver2.prototxt
WEIGHTS1=./ResNet-101-model.caffemodel
#WEIGHTS2=./models/resnet-101-voc_iter_1000.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER1 \
    --weights=$WEIGHTS1 \
    --gpu=0 2>&1 | tee train_coco.log

#$CAFFE_HOME/build/tools/caffe train \
#    --solver=$SOLVER2 \
#    --weights=$WEIGHTS2 \
#    --gpu=3

