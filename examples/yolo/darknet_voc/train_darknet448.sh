#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER=./gnet_region_solver_darknet448_voc.prototxt

#SNAPSHOT
WEIGHTS=./darknet19_448.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=0 2>&1 | tee log/train_darknet_no_reorg.log

