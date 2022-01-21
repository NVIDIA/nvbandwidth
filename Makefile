TARGET = nvbandwidth
CXX = g++
CUDA_CXX = nvcc
CFLAGS = -fopenmp
LIBS = -lboost_program_options -lcuda -lnvidia-ml -lgomp -L/usr/local/cuda/lib64 -lcudart
INCLUDES = /usr/local/cuda/include

.PHONY: default all clean

default: $(TARGET)

all: default

release: CFLAGS += -O3
release: $(TARGET)

debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)

CUDA_OBJECTS = kernels.o
OBJECTS = benchmark.o benchmarks_ce.o benchmarks_sm.o memory_utils.o nvbandwidth.o memcpy.o
HEADERS = $(wildcard *.h)

%.o: %.cu $(HEADERS)
	$(CUDA_CXX) -Xcompiler $(CFLAGS) -c $< -o $@

%.o: %.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c $< -o $@ -I$(INCLUDES)

.PRECIOUS: $(TARGET) $(CUDA_OBJECTS) $(OBJECTS)

$(TARGET): $(CUDA_OBJECTS) $(OBJECTS)
	$(CXX) $(OBJECTS) $(CUDA_OBJECTS) $(LIBS) -o $@

clean:
	-rm -f $(OBJECTS) $(CUDA_OBJECTS) $(TARGET)
