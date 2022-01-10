#define BOOST_TEST_MODULE DotTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <fsmp.cuh>

#include <algorithm>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void dot_wrapper(const std::size_t n, const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ pres)
{
    __shared__ float res;
    dot(n, x, y, &res);
    const std::uint32_t id = threadIdx.x;
    if (id == 0)
    {
        *pres = res;
    }
}

std::random_device rnd_device{};
std::default_random_engine rnd_engine{ rnd_device() };

void dot_test_length(size_t LENGTH)
{
    std::uniform_real_distribution<float> rnd{};
    thrust::host_vector<float> hx(LENGTH, 0.0f);
    thrust::host_vector<float> hy(LENGTH, 0.0f);
    std::vector<float> dot(LENGTH, 0.0f);
    for (size_t i = 0; i < LENGTH; i++)
    {
        hx[i] = rnd(rnd_engine);
        hy[i] = rnd(rnd_engine);
        dot[i] += hx[i] * hy[i];
    }
    float res = std::accumulate(dot.begin(), dot.end(), 0.0f);

    thrust::device_vector<float> dx = hx;
    thrust::device_vector<float> dy = hy;
    thrust::device_vector<float> dres(1, 0.0f);
    dot_wrapper CUDA_KERNEL(1, 1024)(LENGTH, dx.data().get(), dy.data().get(), dres.data().get());
    float ddres = dres[0];
    BOOST_REQUIRE_CLOSE(res, ddres, 1e-3f);
}

BOOST_AUTO_TEST_CASE(dot_test_1024)
{
    dot_test_length(1024);
}

BOOST_AUTO_TEST_CASE(dot_test_128)
{
    dot_test_length(128);
}

__global__ void real_time_wrapper(const std::size_t n, const std::size_t nl, float* __restrict__ pt, const float* __restrict__ tlist)
{
    real_time(n, nl, pt, tlist);
}

BOOST_AUTO_TEST_CASE(real_time_test_fixed)
{
    std::vector<float> hts{ 100, 101, 102, 103, 104, 105, 106, 107, 108 };
    std::vector<float> htlist{ 100, 102, 104, 106, 108 };
    std::vector<float> expect{ 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4 };

    thrust::device_vector<float> dts = hts;
    thrust::device_vector<float> dtlist = htlist;
    real_time_wrapper CUDA_KERNEL(1, 1024)(dts.size(), dtlist.size(), dts.data().get(), dtlist.data().get());

    BOOST_REQUIRE_EQUAL_COLLECTIONS(expect.begin(), expect.end(), dts.begin(), dts.end());
}

BOOST_AUTO_TEST_CASE(real_time_test_random)
{
    constexpr size_t NT = 100;
    constexpr size_t NTLIST = 10;
    thrust::host_vector<float> hts(NT, 0.0);
    thrust::host_vector<float> htlist(NTLIST, 0.0);
    std::uniform_real_distribution<float> rnd{ 100, 200 };
    for (size_t i = 0; i < NT; i++)
    {
        hts[i] = rnd(rnd_engine);
    }
    for (size_t i = 1; i < NTLIST - 1; i++)
    {
        htlist[i] = rnd(rnd_engine);
    }
    htlist[0] = 100;
    htlist[NTLIST - 1] = 200;
    std::sort(hts.begin(), hts.end());
    std::sort(htlist.begin(), htlist.end());

    thrust::device_vector<float> dts = hts;
    thrust::device_vector<float> dtlist = htlist;
    real_time_wrapper CUDA_KERNEL(1, 1024)(NT, NTLIST, dts.data().get(), dtlist.data().get());

    for (size_t i = 0; i < NT; i++)
    {
        float t = hts[i];
        size_t j = 0;
        for (; j < NTLIST; j++)
        {
            if (htlist[j] > t) break;
        }
        hts[i] = j - 1 + (t - htlist[j - 1]) / (htlist[j] - htlist[j - 1]);
    }

    BOOST_REQUIRE_EQUAL_COLLECTIONS(hts.begin(), hts.end(), dts.begin(), dts.end());
}
