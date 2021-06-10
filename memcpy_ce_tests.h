#include "default_constants.h"
#include <string>
#include "nvbw_device.h"

#ifndef _MEMCPY_CE_TEST_H_
#define _MEMCPY_CE_TEST_H_

void launch_HtoD_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount, DeviceFilter filter = alwaysTrueDeviceFilter);

void launch_DtoH_memcpy_bidirectional_CE(const std::string &test_name, unsigned long long size = defaultBufferSize, unsigned long long loopCount = defaultLoopCount, DeviceFilter filter = alwaysTrueDeviceFilter);

#endif // _MEMCPY_CE_TEST_H_
