#include <iostream>
#include <cstdint>

#include <functions.cuh>
#include <kernels.cuh>

using std::cout;
using std::cin;
using std::endl;

void generateFunction(uint16_t *d_out, uint64_t size, uint64_t i0);

uint64_t findMaximum(const uint64_t *data, uint64_t size);

uint64_t generateAndFindMaximum(uint16_t *d_fn, uint64_t *d_buf, uint64_t *h_out, uint64_t size, uint64_t i0);

int main()
{
    const uint64_t size = 1ULL << 31u;

    uint16_t maxN = 1;
    uint64_t maxArg = 2;

    uint16_t *d_fn;
    cudaMalloc(&d_fn, size * sizeof(*d_fn));

    // oh my god this is so wrong
    uint64_t blocks = (size + Kernels::numReduceThreads - 1) / Kernels::numReduceThreads;
    uint64_t *d_buf;
    cudaMalloc(&d_buf, blocks * sizeof(*d_buf));

    auto *h_out = new uint64_t[blocks];

    for (int i = 0; i < 32; i++)
    {
        auto newMax = generateAndFindMaximum(d_fn, d_buf, h_out, size, i * size);
        uint16_t newMaxN = newMax & 0xffffU;
        uint64_t newMaxArg = newMax >> 16U;
        if (newMaxN > maxN)
        {
            maxN = newMaxN;
            maxArg = newMaxArg;
        }
    }

    cudaFree(d_fn);
    cudaFree(d_buf);
    delete [] h_out;

    cout << "Final max = " << maxArg << ", " << maxN << endl;

    return 0;
}

void generateFunction(uint16_t *d_out, uint64_t size, uint64_t i0)
{
    int threads = Kernels::numGenThreads;
    uint64_t blocks = (size + threads - 1) / threads;
    Kernels::genFnKn<<<blocks, threads>>>(d_out, size, i0);
}

uint64_t findMaximum(const uint64_t *data, uint64_t size)
{
    uint64_t max = data[0];
    for (uint64_t i = 0; i < size; i++)
    {
        if ((data[i] & 0xffffu) > (max & 0xffffu))
        {
            max = data[i];
        }
    }
    return max;
}

uint64_t generateAndFindMaximum(uint16_t *d_fn, uint64_t *d_buf, uint64_t *h_out, uint64_t size, uint64_t i0)
{
    cout << "Generating " << size << " values from " << i0 << endl;
    generateFunction(d_fn, size, i0);

//    cudaDeviceSynchronize();
    cout << "Looking for maximum" << endl;

    // oh my god this is so wrong
    uint64_t blocks = (size + Kernels::numReduceThreads - 1) / Kernels::numReduceThreads;

    Kernels::findMax<<<blocks, Kernels::numReduceThreads>>>(d_buf, d_fn, size);

    cudaMemcpy(h_out, d_buf, blocks * sizeof(*d_buf), cudaMemcpyDeviceToHost);

    auto max = findMaximum(h_out, blocks);
    uint16_t maxN = max & 0xffffu;
    uint64_t maxArg = (max >> 16U) + i0;

    cout << "Maximum = " << maxArg << ", " << maxN << endl;

    return (maxArg << 16U) | maxN;
}
