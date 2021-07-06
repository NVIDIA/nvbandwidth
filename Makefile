TARGET = nvbandwidth
CXX = nvcc
CFLAGS =
LIBS = -lboost_program_options -lcuda

.PHONY: default all clean

default: $(TARGET)

all: default

debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)

CUDA_OBJECTS = $(patsubst %.cu, %.o, $(wildcard *.cu))
OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

%.o: %.cu $(HEADERS)
	$(CXX) $(CFLAGS) -c $< -o $@

%.o: %.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(CUDA_OBJECTS) $(OBJECTS)

$(TARGET): $(CUDA_OBJECTS) $(OBJECTS)
	$(CXX) $(CUDA_OBJECTS) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
