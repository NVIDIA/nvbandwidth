#include "options.h"

bool is_dvs()
{
    char envBuf[4];

    // TODO : For now ignoring DVS as cuosGetEnv is simply not worth implementing now
    /*
    if (0 == cuosGetEnv("DVS_ACTIVE", envBuf, 4)) {
        return (0 == strcmp(envBuf, "yes"));
    }
    */

    return false;
}
