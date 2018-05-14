#!/usr/bin/env sh

CAFFE_HOME=../../..

SOLVER=./gnet_region_solver_googlenet.prototxt
#SNAPSHOT=./models/gnet_yolo_region_googlenet_iter_5000.solverstate
WEIGHTS=./bvlc_googlenet.caffemodel

$CAFFE_HOME/build/tools/caffe train \
    --solver=$SOLVER --weights=$WEIGHTS --gpu=1 2>&1 | tee train.log

