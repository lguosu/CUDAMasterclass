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

# Rule for regular CUDA files
%.exe: %.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -o $@ $< $(COMMON_SRC)

# Special rules for dynamic parallelism files
14_dynamic_parallelism.exe: 14_dynamic_parallelism.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $<.o
	$(NVCC) $(NVCC_FLAGS) -dlink $<.o $(COMMON_SRC) -o $@

14_reduction_with_dynamic_parallelism.exe: 14_reduction_with_dynamic_parallelism.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $<.o
	$(NVCC) $(NVCC_FLAGS) -dlink $<.o $(COMMON_SRC) -o $@

test_code.exe: test_code.cu $(COMMON_SRC)
	$(NVCC) $(NVCC_FLAGS) -dc $< -o $<.o
	$(NVCC) $(NVCC_FLAGS) -dlink $<.o $(COMMON_SRC) -o $@

# Clean target
clean:
	rm -f $(EXECUTABLES) *.o

.PHONY: all clean
