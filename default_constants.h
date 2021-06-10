#ifndef _DEFAULT_CONSTANTS_H_
#define _DEFAULT_CONSTANTS_H_

#ifdef DIRECTAMODEL
const unsigned long long defaultLoopCount = 4;
const unsigned long long defaultBufferSize = 1024 * 1024 * 2; // 2MB
const unsigned int defaultAverageLoopCount = 1;

const unsigned long long defaultLoopCount_dvs = 4;
const unsigned long long defaultBufferSize_dvs = 1024 * 1024 * 2; // 64MB
const unsigned int defaultAverageLoopCount_dvs = 1;

#else
const unsigned long long defaultLoopCount = 16;
const unsigned long long defaultBufferSize = 1024 * 1024 * 64; // 64MB
const unsigned int defaultAverageLoopCount = 3;

const unsigned long long defaultLoopCount_dvs = 4;
const unsigned long long defaultBufferSize_dvs = 1024 * 1024 * 32; // 32MB
const unsigned int defaultAverageLoopCount_dvs = 3;
#endif

const unsigned long long defaultWddmPacketSchedulingSizeWAR = 1024 * 1024 * 128; // 128MB

#endif // _DEFAULT_CONSTANTS_H_
