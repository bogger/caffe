#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/bvlc_alexnet \
./build/tools/caffe train \
    --solver=models/bvlc_alexnet/solver.prototxt \
    --gpu=2
