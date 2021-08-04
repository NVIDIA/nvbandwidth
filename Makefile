TARGET = nvbandwidth
CXX = nvcc
CFLAGS =
LIBS = -lboost_program_options -lcuda

.PHONY: default all clean

default: $(TARGET)

all: default

release: CFLAGS += -O3
release: $(TARGET)

debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)

CUDA_OBJECTS = memcpy_kernel.o
OBJECTS = benchmarks_ce.o benchmarks_sm.o memory_utils.o nvbandwidth.o
HEADERS = $(wildcard *.h)

%.o: %.cu $(HEADERS)
	$(CXX) $(CFLAGS) -c $< -o $@

%.o: %.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(CUDA_OBJECTS) $(OBJECTS)

$(TARGET): $(CUDA_OBJECTS) $(OBJECTS)
	$(CXX) $(CUDA_OBJECTS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f $(OBJECTS) $(CUDA_OBJECTS) $(TARGET)
