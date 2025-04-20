#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_common.cuh"

// Template kernel for both float and double computations
template<typename T>
__global__ void lots_of_compute(T *inputs, int N, size_t niters, T *outputs)
{
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t nthreads = gridDim.x * blockDim.x;

    for (; tid < N; tid += nthreads)
    {
        size_t iter;
        T val = inputs[tid];

        for (iter = 0; iter < niters; iter++)
        {
            val = (val + T(5.0)) - T(101.0);
            val = (val / T(3.0)) + T(102.0);
            val = (val + T(1.07)) - T(103.0);
            val = (val / T(1.037)) + T(104.0);
            val = (val + T(3.00)) - T(105.0);
            val = (val / T(0.22)) + T(106.0);
        }

        outputs[tid] = val;
    }
}

// Template function for running tests
template<typename T>
static void run_test(size_t N, int niters, int blocksPerGrid, int threadsPerBlock,
    long* to_device_clock_cyl, long* kernel_clock_cyl, long* from_device_clock_cyl,
    T* sample, int sampleLength)
{
    int i;
    T *h_inputs, *h_outputs;
    T *d_inputs, *d_outputs;

    h_inputs = (T*)malloc(sizeof(T) * N);
    h_outputs = (T*)malloc(sizeof(T) * N);
    gpuErrchk(cudaMalloc((void**)&d_inputs, sizeof(T) * N));
    gpuErrchk(cudaMalloc((void**)&d_outputs, sizeof(T) * N));

    for (i = 0; i < N; i++)
    {
        h_inputs[i] = (T)i;
    }

    clock_t ops_start, ops_end;

    ops_start = clock();
    gpuErrchk(cudaMemcpy(d_inputs, h_inputs, sizeof(T) * N, cudaMemcpyHostToDevice));
    ops_end = clock();
    *to_device_clock_cyl = ops_end - ops_start;

    ops_start = clock();
    lots_of_compute<T><<<blocksPerGrid, threadsPerBlock>>>(d_inputs, N, niters, d_outputs);
    gpuErrchk(cudaDeviceSynchronize());
    ops_end = clock();
    *kernel_clock_cyl = ops_end - ops_start;

    ops_start = clock();
    gpuErrchk(cudaMemcpy(h_outputs, d_outputs, sizeof(T) * N, cudaMemcpyDeviceToHost));
    ops_end = clock();
    *from_device_clock_cyl = ops_end - ops_start;

    for (i = 0; i < sampleLength; i++)
    {
        sample[i] = h_outputs[i];
    }

    gpuErrchk(cudaFree(d_inputs));
    gpuErrchk(cudaFree(d_outputs));
    free(h_inputs);
    free(h_outputs);
}

int main(int argc, char **argv)
{
    int i;
    double meanFloatToDeviceTime, meanFloatKernelTime, meanFloatFromDeviceTime;
    double meanDoubleToDeviceTime, meanDoubleKernelTime,
        meanDoubleFromDeviceTime;
    struct cudaDeviceProp deviceProperties;
    size_t totalMem, freeMem;
    float *floatSample;
    double *doubleSample;
    int sampleLength = 10;
    int nRuns = 5;
    int nKernelIters = 20;

    meanFloatToDeviceTime = meanFloatKernelTime = meanFloatFromDeviceTime = 0.0;
    meanDoubleToDeviceTime = meanDoubleKernelTime =
        meanDoubleFromDeviceTime = 0.0;

    gpuErrchk(cudaMemGetInfo(&freeMem, &totalMem));
    gpuErrchk(cudaGetDeviceProperties(&deviceProperties, 0));

    size_t N = (freeMem * 0.9 / 2) / sizeof(double);
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    if (blocksPerGrid > deviceProperties.maxGridSize[0])
    {
        blocksPerGrid = deviceProperties.maxGridSize[0];
    }

    printf("Running %d blocks with %d threads/block over %lu elements\n",
        blocksPerGrid, threadsPerBlock, N);

    floatSample = (float *)malloc(sizeof(float) * sampleLength);
    doubleSample = (double *)malloc(sizeof(double) * sampleLength);

    for (i = 0; i < nRuns; i++)
    {
        long toDeviceTime, kernelTime, fromDeviceTime;

        run_test<float>(N, nKernelIters, blocksPerGrid, threadsPerBlock,
            &toDeviceTime, &kernelTime, &fromDeviceTime,
            floatSample, sampleLength);
        meanFloatToDeviceTime += toDeviceTime;
        meanFloatKernelTime += kernelTime;
        meanFloatFromDeviceTime += fromDeviceTime;

        run_test<double>(N, nKernelIters, blocksPerGrid, threadsPerBlock,
            &toDeviceTime, &kernelTime, &fromDeviceTime,
            doubleSample, sampleLength);
        meanDoubleToDeviceTime += toDeviceTime;
        meanDoubleKernelTime += kernelTime;
        meanDoubleFromDeviceTime += fromDeviceTime;
    }

    meanFloatToDeviceTime /= nRuns;
    meanFloatKernelTime /= nRuns;
    meanFloatFromDeviceTime /= nRuns;
    meanDoubleToDeviceTime /= nRuns;
    meanDoubleKernelTime /= nRuns;
    meanDoubleFromDeviceTime /= nRuns;

    meanFloatToDeviceTime /= CLOCKS_PER_SEC;
    meanFloatKernelTime /= CLOCKS_PER_SEC;
    meanFloatFromDeviceTime /= CLOCKS_PER_SEC;
    meanDoubleToDeviceTime /= CLOCKS_PER_SEC;
    meanDoubleKernelTime /= CLOCKS_PER_SEC;
    meanDoubleFromDeviceTime /= CLOCKS_PER_SEC;

    printf("For single-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s\n", meanFloatToDeviceTime);
    printf("  Kernel execution: %f s\n", meanFloatKernelTime);
    printf("  Copy from device: %f s\n", meanFloatFromDeviceTime);
    printf("For double-precision floating point, mean times for:\n");
    printf("  Copy to device:   %f s (%.2fx slower than single-precision)\n",
        meanDoubleToDeviceTime,
        meanDoubleToDeviceTime / meanFloatToDeviceTime);
    printf("  Kernel execution: %f s (%.2fx slower than single-precision)\n",
        meanDoubleKernelTime,
        meanDoubleKernelTime / meanFloatKernelTime);
    printf("  Copy from device: %f s (%.2fx slower than single-precision)\n",
        meanDoubleFromDeviceTime,
        meanDoubleFromDeviceTime / meanFloatFromDeviceTime);

    return 0;
}
