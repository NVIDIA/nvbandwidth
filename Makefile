CXX = nvcc

HEADERS = mem_allocator.h memcpy_ce_tests.h default_constants.h spinKernel.h nvbw_os.h nvbw_device.h stats.h options.h mem_pattern.h

default: NVBandwidth

mem_pattern.o: mem_pattern.cpp $(HEADERS)
	$(CXX) -lcuda -c mem_pattern.cpp -o mem_pattern.o
options.o: options.cpp $(HEADERS)
	$(CXX) -lcuda -c options.cpp -o options.o
nvbandwidth.o: nvbandwidth.cpp $(HEADERS)
	$(CXX) -lcuda -c nvbandwidth.cpp -o nvbandwidth.o
mem_allocator.o: mem_allocator.cpp $(HEADERS)
	$(CXX) -lcuda -c mem_allocator.cpp -o mem_allocator.o
spinKernel.o: spinKernel.cpp $(HEADERS)
	$(CXX) -lcuda -c spinKernel.cpp -o spinKernel.o
nvbw_os.o: nvbw_os.cpp $(HEADERS)
	$(CXX) -lcuda -c nvbw_os.cpp -o nvbw_os.o
nvbw_device.o: nvbw_device.cpp $(HEADERS)
	$(CXX) -lcuda -c nvbw_device.cpp -o nvbw_device.o
memcpy_ce_tests.o: memcpy_ce_tests.cpp $(HEADERS)
	$(CXX) -lcuda -c memcpy_ce_tests.cpp -o memcpy_ce_tests.o
NVBandwidth: nvbw_device.o nvbw_os.o mem_pattern.o options.o nvbandwidth.o mem_allocator.o spinKernel.o memcpy_ce_tests.o
	$(CXX) -lboost_program_options -lcuda nvbw_os.o nvbw_device.o mem_pattern.o options.o nvbandwidth.o mem_allocator.o spinKernel.o memcpy_ce_tests.o -o NVBandwidth

clean:
	-rm -f *.o
	-rm -f dfontaine/*.o
	-rm -f dfontaine/testsuite/*.o
	-rm -f NVBandwidth
