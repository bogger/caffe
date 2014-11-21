#!/usr/bin/env sh

GOOGLE_LOG_DIR=models/classitection/log \
/usr/local/openmpi/bin/mpirun -n 3 build/tools/caffe train \
    --solver=models/classitection/solver.prototxt \
