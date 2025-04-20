# Compiler settings
NVCC = nvcc
NVCC_FLAGS = --std=c++20 -I../common
COMMON_SRC = ../common/common.cpp

# Find all CUDA source files
CUDA_SRCS := $(wildcard */*.cu)
# Convert source names to executable names
EXECUTABLES := $(CUDA_SRCS:.cu=.exe)

# Default target
all: $(EXECUTABLES)

# Rule for regular CUDA files
%.exe: %.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $^

# Clean target
clean:
	rm -f $(EXECUTABLES)

.PHONY: all clean
