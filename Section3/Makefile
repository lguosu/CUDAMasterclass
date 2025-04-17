# Compiler settings
NVCC = nvcc
NVCC_FLAGS = -I../common
COMMON_SRC = ../common/common.cpp

# Find all CUDA source files
CUDA_SRCS := $(wildcard *.cu)
# Convert source names to executable names
EXECUTABLES := $(CUDA_SRCS:.cu=.exe)

# Default target
all: $(EXECUTABLES)

# Rule to build executables
%.exe: %.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

6_misaligned_read.exe: 6_misaligned_read.cu
	# -Xptxas -dlcm=ca - These flags control the L1 cache behavior:
	#	-Xptxas tells NVCC to pass the following options to the PTX assembler
	# 	-dlcm=ca sets the L1 cache configuration:
	# 	c stands for "cache" - enables L1 cache
	# 	a stands for "all" - allows both loads and stores to use the L1 cache
	$(NVCC) $(NVCC_FLAGS) -Xptxas -dlcm=ca -o $@ $^

# Clean target
clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean
