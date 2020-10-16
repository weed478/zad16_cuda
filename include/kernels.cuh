#ifndef FNMAX_KERNELS_CUH
#define FNMAX_KERNELS_CUH

#include <cstdint>

namespace Kernels
{
    const int numGenThreads = 128;
    const int numReduceThreads = 256;

    __global__
    void genFnKn(uint16_t *out, uint64_t size, uint64_t i0);

    __global__
    void findMax(uint64_t *out, const uint16_t *in, uint64_t size);
}

#endif //FNMAX_KERNELS_CUH
