#!/bin/bash

# Do not use -g or -G flag since it will disable the optimization
# and generate a lot of debug information

cd ..
nvcc -c common.cpp -o common.obj
cd -
nvcc -link ../common/common.obj -o profiling_test.out 7_sum_array.cu -I../common

./profiling_test.out

nvprof ./profiling_test.out
