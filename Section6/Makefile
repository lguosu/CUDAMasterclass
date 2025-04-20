# Compiler settings
NVCC = nvcc
NVCC_FLAGS = --std=c++20 -I../common
COMMON_SRC = ../common/common.cpp

# Find all CUDA source files
CUDA_SRCS := $(wildcard *.cu)
# Convert source names to executable names
EXECUTABLES := $(CUDA_SRCS:.cu=.exe)

INTRINSIC_SRCS := $(wildcard 3_standard_intrinsic*.cu)
PTX_FILES := $(INTRINSIC_SRCS:.cu=.ptx)

# Default target
all: $(EXECUTABLES) $(PTX_FILES)

# Rule for regular CUDA files
%.exe: %.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

%.ptx: %.cu
	$(NVCC) $(NVCC_FLAGS) --ptx -o $@ $^
# Clean target
clean:
	rm -f $(EXECUTABLES) *.o

.PHONY: all clean
