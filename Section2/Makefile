# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -I../common
COMMON_SRC = ../common/common.cpp
DYNAMIC_PARALLELISM_FLAGS = -arch=sm_80 -rdc=true

# Find all CUDA source files
CUDA_SRCS := $(wildcard *.cu)
# Convert source names to executable names
EXECUTABLES := $(CUDA_SRCS:.cu=.exe)

# Default target
all: $(EXECUTABLES)

# Rule for regular CUDA files
%.exe: %.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Special rules for dynamic parallelism files
14_dynamic_parallelism.exe: 14_dynamic_parallelism.cu
	$(NVCC) $(NVCC_FLAGS) $(DYNAMIC_PARALLELISM_FLAGS) -o $@ $^

14_reduction_with_dynamic_parallelism.exe: 14_reduction_with_dynamic_parallelism.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) $(DYNAMIC_PARALLELISM_FLAGS) -o $@ $^

test_code.exe: test_code.cu
	$(NVCC) $(NVCC_FLAGS) $(DYNAMIC_PARALLELISM_FLAGS) -o $@ $^

# Clean target
clean:
	rm -f $(EXECUTABLES) *.o

.PHONY: all clean
