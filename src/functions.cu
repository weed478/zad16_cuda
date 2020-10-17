#include <functions.cuh>

__device__
uint16_t getNForSeries(uint64_t a)
{
    uint16_t n = 0;
    while (a != 1)
    {
        if (a % 2)
        {
            a = (3 * a + 1) / 2;
            n += 2;
        }
        else
        {
            a = a / 2;
            n += 1;
        }
    }
    return n;
}