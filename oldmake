CXX = nvcc
CFLAGS = -g
HEADERS = mem_allocator.h memcpy_ce_tests.h default_constants.h spinKernel.h device.h stats.h options.h mem_pattern.h

# TODO : Include debug (-d) flag for develop builds (release builds to be added)

default: nvbandwidth

mem_pattern.o: mem_pattern.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c mem_pattern.cpp -o mem_pattern.o
options.o: options.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c options.cpp -o options.o
nvbandwidth.o: nvbandwidth.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c nvbandwidth.cpp -o nvbandwidth.o
mem_allocator.o: mem_allocator.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c mem_allocator.cpp -o mem_allocator.o
spinKernel.o: spinKernel.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c spinKernel.cpp -o spinKernel.o
device.o: device.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c device.cpp -o device.o
memcpy_ce_tests.o: memcpy_ce_tests.cpp $(HEADERS)
	$(CXX) $(CFLAGS) -c memcpy_ce_tests.cpp -o memcpy_ce_tests.o
nvbandwidth: device.o mem_pattern.o options.o nvbandwidth.o mem_allocator.o spinKernel.o memcpy_ce_tests.o
	$(CXX) $(CFLAGS) -lboost_program_options -lcuda device.o mem_pattern.o options.o nvbandwidth.o mem_allocator.o spinKernel.o memcpy_ce_tests.o -o nvbandwidth

clean:
	-rm -f *.o
	-rm -f nvbandwidth
