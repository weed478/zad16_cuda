#include <functions.cuh>

__device__
uint64_t nextSeries(uint64_t a)
{
//    if (a % 2)
//        return (3 * a + 1);
//    else
//        return a / 2;

    return (a % 2) * (3 * a + 1) + (1 - a % 2) * a / 2;
}

__device__
uint16_t getNForSeries(uint64_t a)
{
    uint16_t n = 0;
    while (a != 1)
    {
        a = nextSeries(a);
        n++;
    }
    return n;
}