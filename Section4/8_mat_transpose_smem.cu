#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "common.h"
#include "cuda_common.cuh"

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#define BDIMX 64
#define BDIMY 8
#define IPAD 2 // for 64-bit bank width shared memory

__global__ void transpose_read_raw_write_column_benchmark(int * mat, 
	int* transpose, int nx, int ny)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	if (ix < nx && iy < ny)
	{
		//read by row, write by col
		transpose[ix * ny + iy] = mat[iy * nx + ix];
	}
}

__global__ void transpose_smem(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMY][BDIMX];

	//input index
	int ix, iy, in_index;

	//output index
	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

	//ix and iy calculation for input index
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;

	//input index
	in_index = iy * nx + ix;

	//1D index calculation fro shared memory
	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;

	//col major row and col index calcuation
	i_row = _1d_index / blockDim.y;
	i_col = _1d_index % blockDim.y;

	//coordinate for transpose matrix
	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = blockIdx.x * blockDim.x + i_row;

	//output array access in row major format
	out_index = out_iy * ny + out_ix;

	if (ix < nx && iy < ny)
	{
		//load from in array in row major and store to shared memory in row major
		tile[threadIdx.y][threadIdx.x] = in[in_index];

		//wait untill all the threads load the values
		__syncthreads();

		out[out_index] = tile[i_col][i_row];
	}
}

__global__ void transpose_smem_pad(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMY][BDIMX + IPAD];

	//input index
	int ix, iy, in_index;

	//output index
	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

	//ix and iy calculation for input index
	ix = blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;

	//input index
	in_index = iy * nx + ix;

	//1D index calculation fro shared memory
	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;

	//col major row and col index calcuation
	i_row = _1d_index / blockDim.y;
	i_col = _1d_index % blockDim.y;

	//coordinate for transpose matrix
	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = blockIdx.x * blockDim.x + i_row;

	//output array access in row major format
	out_index = out_iy * ny + out_ix;

	if (ix < nx && iy < ny)
	{
		//load from in array in row major and store to shared memory in row major
		tile[threadIdx.y][threadIdx.x] = in[in_index];

		//wait untill all the threads load the values
		__syncthreads();

		out[out_index] = tile[i_col][i_row];
	}
}

__global__ void transpose_smem_pad_unrolling(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMY * (2 * BDIMX + IPAD)];

	//input index
	int ix, iy, in_index;

	//output index
	int i_row, i_col, _1d_index, out_ix, out_iy, out_index;

	//ix and iy calculation for input index
	ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	iy = blockDim.y * blockIdx.y + threadIdx.y;

	//input index
	in_index = iy * nx + ix;

	//1D index calculation fro shared memory
	_1d_index = threadIdx.y * blockDim.x + threadIdx.x;

	//col major row and col index calcuation
	i_row = _1d_index / blockDim.y;
	i_col = _1d_index % blockDim.y;

	//coordinate for transpose matrix
	out_ix = blockIdx.y * blockDim.y + i_col;
	out_iy = 2 * blockIdx.x * blockDim.x + i_row;

	//output array access in row major format
	out_index = out_iy * ny + out_ix;

	if (ix < nx && iy < ny)
	{
		int row_idx = threadIdx.y * (2 * blockDim.x + IPAD) + threadIdx.x;

		//load from in array in row major and store to shared memory in row major
		tile[row_idx] = in[in_index];
		tile[row_idx+ BDIMX] = in[in_index + BDIMX];

		//wait untill all the threads load the values
		__syncthreads();

		int col_idx = i_col * (2 * blockDim.x + IPAD) + i_row;

		out[out_index] = tile[col_idx];
		out[out_index + ny* BDIMX] = tile[col_idx + BDIMX];
	}
}

__global__ void transpose_smem_diagonal(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[BDIMX][BDIMY];

	//ix and iy calculation for input index
	int ix = blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	//input index
	int in_index = iy * nx + ix;

	//output array access in row major format
	int out_index = ix * ny + iy;

	if (ix < nx && iy < ny)
	{
		int diag_row = threadIdx.x;
		int diag_col = (threadIdx.x + threadIdx.y) % blockDim.y;
		tile[diag_row][diag_col] = in[in_index];

		//wait untill all the threads load the values
		__syncthreads();

		out[out_index] = tile[diag_row][diag_col];
	}
}

__global__ void transpose_smem_diagonal_unrolling(int * in, int* out, int nx, int ny)
{
	__shared__ int tile[2 * BDIMX][BDIMY];

	//ix and iy calculation for input index
	int ix = 2 * blockDim.x * blockIdx.x + threadIdx.x;
	int iy = blockDim.y * blockIdx.y + threadIdx.y;

	//input index
	int in_index1 = iy * nx + ix;
	int in_index2 = in_index1 + BDIMX;

	//output array access in row major format
	int out_index1 = ix * ny + iy;
	int out_index2 = (ix + BDIMX) * ny + iy;

	if (ix < nx && iy < ny)
	{
		int diag_row1 = threadIdx.x;
		int diag_col1 = (threadIdx.x + threadIdx.y) % blockDim.y;
		tile[diag_row1][diag_col1] = in[in_index1];

		int diag_row2 = threadIdx.x + BDIMX;
		int diag_col2 = (threadIdx.x + BDIMX + threadIdx.y) % blockDim.y;
		tile[diag_row2][diag_col2] = in[in_index2];

		//wait untill all the threads load the values
		__syncthreads();

		out[out_index1] = tile[diag_row1][diag_col1];
		out[out_index2] = tile[diag_row2][diag_col2];
	}
}

int main(int argc, char** argv)
{
	//default values for variabless
	int nx = 1024;
	int ny = 1024;
	int block_x = BDIMX;
	int block_y = BDIMY;
	int kernel_num = 0;

	//set the variable based on arguments
	if (argc > 1)
		nx = 1 << atoi(argv[1]);
	if (argc > 2)
		ny = 1 << atoi(argv[2]);
	if (argc > 3)
		block_x = 1 << atoi(argv[3]);
	if (argc > 4)
		block_y = 1 <<atoi(argv[4]);

	int size = nx * ny;
	int byte_size = sizeof(int*) * size;

	printf("Matrix transpose for %d X % d matrix with block size %d X %d \n",nx,ny,block_x,block_y);

	int * h_mat_array = (int*)malloc(byte_size);
	int * h_trans_array = (int*)malloc(byte_size);
	int * h_ref = (int*)malloc(byte_size);

	initialize(h_mat_array,size ,INIT_ONE_TO_TEN);

	//matirx transpose in CPU
	mat_transpose_cpu(h_mat_array, h_trans_array, nx, ny);

	int * d_mat_array, *d_trans_array;
	
	gpuErrchk(cudaMalloc((void**)&d_mat_array, byte_size));
	gpuErrchk(cudaMalloc((void**)&d_trans_array, byte_size));

	gpuErrchk(cudaMemcpy(d_mat_array, h_mat_array, byte_size, cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemset(d_trans_array, 0, byte_size));

	dim3 blocks(block_x, block_y);
	dim3 grid(nx/block_x, ny/block_y);
	dim3 unrolling_grid(grid.x/2, grid.y);

	// Create a lambda that captures the common parameters
	auto execute_kernel = [=](const char* kernel_name, auto kernel, dim3 grid, dim3 blocks) {
		printf("Launching %s kernel\n", kernel_name);
		
		// Create CUDA events for timing
		cudaEvent_t start, stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		cudaMemset(d_trans_array, 0, byte_size);
		
		// Record start event
		cudaEventRecord(start);
		
		kernel<<<grid, blocks>>>(d_mat_array, d_trans_array, nx, ny);
		gpuErrchk(cudaDeviceSynchronize());
		
		// Record stop event
		cudaEventRecord(stop);
		cudaEventSynchronize(stop);
		
		// Calculate elapsed time
		float milliseconds = 0;
		cudaEventElapsedTime(&milliseconds, start, stop);
		printf("Kernel execution time: %f ms\n", milliseconds);
		
		// Clean up events
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
		
		gpuErrchk(cudaMemcpy(h_ref, d_trans_array, byte_size, cudaMemcpyDeviceToHost));
		compare_arrays(h_ref, h_trans_array, size);
	};

	// Execute each kernel using the lambda
	execute_kernel("smem", transpose_smem, grid, blocks);
	execute_kernel("benchmark", transpose_read_raw_write_column_benchmark, grid, blocks);
	execute_kernel("smem padding", transpose_smem_pad, grid, blocks);
	execute_kernel("smem padding and unrolling", transpose_smem_pad_unrolling, unrolling_grid, blocks);
	execute_kernel("smem diagonal indexing", transpose_smem_diagonal, grid, blocks);
	execute_kernel("smem diagonal indexing and unrolling", transpose_smem_diagonal_unrolling, unrolling_grid, blocks);

	cudaFree(d_trans_array);
	cudaFree(d_mat_array);
	free(h_ref);
	free(h_trans_array);
	free(h_mat_array);

	gpuErrchk(cudaDeviceReset());
	return EXIT_SUCCESS;
}