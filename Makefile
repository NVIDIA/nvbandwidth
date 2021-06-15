TARGET = nvbandwidth
CXX = nvcc
CFLAGS =
LIBS = -lboost_program_options -lcuda

.PHONY: default all clean

default: $(TARGET)

all: default

debug: CFLAGS += -DDEBUG -g
debug: $(TARGET)

OBJECTS = $(patsubst %.cpp, %.o, $(wildcard *.cpp))
HEADERS = $(wildcard *.h)

%.o: %.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c $< -o $@

.PRECIOUS: $(TARGET) $(OBJECTS)

$(TARGET): $(OBJECTS)
	$(CXX) $(OBJECTS) $(LIBS) -o $@

clean:
	-rm -f *.o
	-rm -f $(TARGET)
