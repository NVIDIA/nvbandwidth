
#ifndef _OPTIONS_H_
#define _OPTIONS_H_

#include <vector>
#include <string>

extern unsigned long long bufferSize;
extern unsigned int averageLoopCount;
extern bool noStridingTests;
extern bool noCheckBandwidth;
extern int pairingMode;
extern bool noTraffic;
extern bool isRandom;
extern unsigned int startValue;
extern unsigned int endValue;
extern unsigned int loopCount;
extern std::string ipc_key;
extern std::vector<int> num_threads_per_sm;
extern std::vector<int> element_sizes;
extern bool skip_verif;
extern bool fullNumaTest;
extern bool doFullSweep;
extern bool disableP2P;
extern bool enableCECopyStartMarker;

bool is_dvs();

#endif // _OPTIONS_H_
