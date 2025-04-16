#!/bin/bash

# Do not use -g or -G flag since it will disable the optimization
# and generate a lot of debug information

cd ../common
nvcc -c common.cpp -o common.o
cd -
nvcc -link ../common/common.o -o profiling_test.exe 7_sum_array.cu -I../common

./profiling_test.exe

# nvprof is not supported on devices with compute capability 8.0 and higher.
# Use NVIDIA Nsight Systems for GPU tracing and CPU sampling and NVIDIA Nsight Compute for GPU profiling.
# Refer https://developer.nvidia.com/tools-overview for more details.
# For command line tool, use ncu command.
ncu ./profiling_test.exe
ncu --metrics gld_efficiency,sm_efficiency,achieved_occupancy profiling_test.exe 1 25 20 8 2
