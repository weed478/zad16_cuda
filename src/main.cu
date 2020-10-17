#include <iostream>
#include <cstdint>

#include <functions.cuh>
#include <kernels.cuh>

using std::cout;
using std::cin;
using std::endl;

void generateFunction(uint16_t *d_out, uint64_t size, uint64_t i0);

uint64_t findMaximum(const uint64_t *data, uint64_t size);

uint64_t generateAndFindMaximum(uint16_t *d_fn, uint64_t *d_buf, uint64_t size, uint64_t i0);

// max in 2^31 * 1024: 2156795915823, 1419

int main()
{
    size_t freeMem = 0;
    cudaMemGetInfo(&freeMem, nullptr);

    freeMem -= freeMem / 10;
    cout << "Using " << freeMem / (1u << 20u) << " MB" << endl;

    const uint64_t size = freeMem / 2;

    uint16_t maxN = 1;
    uint64_t maxArg = 2;

    uint16_t *d_fn;
    cudaMalloc(&d_fn, size * sizeof(*d_fn));

    uint64_t blocks = ((size + 2ull - 1ull) / 2ull + Kernels::numReduceThreads - 1ull) / Kernels::numReduceThreads;
    uint64_t *d_buf;
    cudaMallocHost(&d_buf, blocks * sizeof(*d_buf));

    for (int i = 0; i < 4; i++)
    {
        auto newMax = generateAndFindMaximum(d_fn, d_buf, size, i * size);
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

    cout << "Final max = " << maxArg << ", " << maxN << endl;

    return 0;
}

void generateFunction(uint16_t *d_out, uint64_t size, uint64_t i0)
{
    int threads = Kernels::numGenThreads;
    uint64_t blocks = (size + threads - 1ull) / threads;
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

uint64_t generateAndFindMaximum(uint16_t *d_fn, uint64_t *d_buf, uint64_t size, uint64_t i0)
{
    cout << "Generating " << size << " values from " << i0 << endl;
    generateFunction(d_fn, size, i0);

    uint64_t blocks = ((size + 2ull - 1ull) / 2ull + Kernels::numReduceThreads - 1ull) / Kernels::numReduceThreads;
    Kernels::findMax<<<blocks, Kernels::numReduceThreads>>>(d_buf, d_fn, size);

    cudaDeviceSynchronize();
    auto max = findMaximum(d_buf, blocks);
    uint16_t maxN = max & 0xffffu;
    uint64_t maxArg = (max >> 16U) + i0;

    cout << "Maximum = " << maxArg << ", " << maxN << endl;

    return (maxArg << 16U) | maxN;
}
