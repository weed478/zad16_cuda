#include <kernels.cuh>
#include <functions.cuh>

__global__
void Kernels::genFnKn(uint16_t *out, uint64_t size, uint64_t i0)
{
    uint64_t i = (uint64_t)blockIdx.x * (uint64_t)blockDim.x + (uint64_t)threadIdx.x;
    if (1 < i && i < size)
        out[i] = getNForSeries(i0 + i);
}

__global__ void Kernels::findMax(uint64_t *out, const uint16_t *in, uint64_t size)
{
    __shared__ uint64_t sdata[numReduceThreads];

    const int stride = numReduceThreads;

    unsigned int t = threadIdx.x;
    uint64_t i = (uint64_t)blockIdx.x * (uint64_t)blockDim.x * 2 + (uint64_t)threadIdx.x;

    if (i < size)
    {
        if (i + stride < size && in[i + stride] > in[i])
            sdata[t] = ((i + stride) << 16u) | in[i + stride];
        else
            sdata[t] = (i << 16u) | in[i];
    }

    __syncthreads();

    for (unsigned int s = stride / 2; s > 0; s >>= 1u)
    {
        if (t < s)
        {
            if ((sdata[t] & 0xffffu) > (sdata[t + s] & 0xffffu))
                sdata[t] = sdata[t];
            else
                sdata[t] = sdata[t + s];
        }
        __syncthreads();
    }

    if (t == 0)
        out[blockIdx.x] = sdata[0];
}


