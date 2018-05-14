#!/usr/bin/env sh

CAFFE_HOME=../../..

PROTO=./ResNet-101-test.prototxt
ITER=2000
GPU_ID=2

num=1
for i in `seq $num`;
do

MODEL="./models/final.resnet-101-coco_iter_"
MODEL=$MODEL$(expr $i \* 150000)
MODEL=$MODEL".caffemodel"
echo $MODEL

$CAFFE_HOME/build/tools/test_detection \
    --model=$PROTO --iterations=$ITER \
    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=80 --side=13
done

#MODEL="./models/resnet-101-voc_iter_150000.caffemodel"
#$CAFFE_HOME/build/tools/test_detection \
#    --model=$PROTO --iterations=$ITER \
#    --weights=$MODEL --gpu=$GPU_ID --objects=5 --classes=20 --side=13
