#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./ResNet-101-test.prototxt
ITER=500
GPU_ID=2

num=60
#for i in `seq $num`;
#do

#MODEL="./models/resnet-101-voc_iter_"
#MODEL=$MODEL$(expr $i \* 4000)
#MODEL=$MODEL".caffemodel"

MODEL="./models/final.resnet-101-voc_iter_10w.caffemodel"

echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=20 --side=13
#done

#MODEL="./models/resnet-101-voc_iter_150000.caffemodel"
#$CAFFE_HOME/build/tools/test_detection \
#    --model=$PROTO --iterations=$ITER \
#    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=20 --side=13
