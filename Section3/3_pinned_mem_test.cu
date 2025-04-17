#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <stdlib.h>
#include <chrono>

// Function to measure memory transfer time
float measureTransferTime(float* src, float* dst, size_t nbytes, cudaMemcpyKind kind) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);
	cudaMemcpy(dst, src, nbytes, kind);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);  // We need this to ensure events are ready to be read
	
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	
	return milliseconds;
}

int main(int argc, char **argv) 
{   								
	// memory size   128 MBs
	int isize = 1<<25;   
	int nbytes = isize * sizeof(float);
											
	printf("Memory size: %d MB\n", nbytes / (1024 * 1024));
											
	// Warm-up run to avoid initialization overhead
	{
		float *temp_pinned;
		float *temp_device;
		cudaMallocHost((float **)&temp_pinned, nbytes);
		cudaMalloc((float **)&temp_device, nbytes);
		cudaMemcpy(temp_device, temp_pinned, nbytes, cudaMemcpyHostToDevice);
		cudaFreeHost(temp_pinned);
		cudaFree(temp_device);
	}
											
	// Allocate pinned memory
	float *h_pinned;
	cudaMallocHost((float **)&h_pinned, nbytes);
											
	// Allocate regular (unpinned) memory
	float *h_unpinned = (float *)malloc(nbytes);
											
	// Allocate device memory   
	float *d_a; 
	cudaMalloc((float **)&d_a, nbytes);
									
	// Initialize both host memories
	for(int i=0; i<isize; i++) {
		h_pinned[i] = 7;
		h_unpinned[i] = 7;
	}
									
	// Measure pinned memory transfer
	printf("\nPinned Memory Transfer:\n");
	float pinned_to_device = measureTransferTime(h_pinned, d_a, nbytes, cudaMemcpyHostToDevice);
	float pinned_from_device = measureTransferTime(d_a, h_pinned, nbytes, cudaMemcpyDeviceToHost);
	printf("Host to Device: %.3f ms\n", pinned_to_device);
	printf("Device to Host: %.3f ms\n", pinned_from_device);
	printf("Total Transfer Time: %.3f ms\n", pinned_to_device + pinned_from_device);
									
	// Measure unpinned memory transfer
	printf("\nUnpinned Memory Transfer:\n");
	float unpinned_to_device = measureTransferTime(h_unpinned, d_a, nbytes, cudaMemcpyHostToDevice);
	float unpinned_from_device = measureTransferTime(d_a, h_unpinned, nbytes, cudaMemcpyDeviceToHost);
	printf("Host to Device: %.3f ms\n", unpinned_to_device);
	printf("Device to Host: %.3f ms\n", unpinned_from_device);
	printf("Total Transfer Time: %.3f ms\n", unpinned_to_device + unpinned_from_device);
									
	// Calculate speedup
	float total_pinned = pinned_to_device + pinned_from_device;
	float total_unpinned = unpinned_to_device + unpinned_from_device;
	float speedup = total_unpinned / total_pinned;
	printf("\nPinned memory is %.2fx faster than unpinned memory\n", speedup);
									
	// Free memory   
	cudaFree(d_a);
	cudaFreeHost(h_pinned);
	free(h_unpinned);
									
	// Reset device    
	cudaDeviceReset();   
	return EXIT_SUCCESS;
}