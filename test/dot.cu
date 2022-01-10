#define BOOST_TEST_MODULE DotTest
#include <boost/test/unit_test.hpp>

#include <fsmp.cuh>

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

BOOST_AUTO_TEST_CASE(dot_test)
{
    constexpr size_t LENGTH = 1024;
    std::uniform_real_distribution<float> rnd{};
    thrust::host_vector<float> hx(LENGTH, 0.0f);
    thrust::host_vector<float> hy(LENGTH, 0.0f);
    float res = 0.0;
    for (size_t i = 0; i < LENGTH; i++)
    {
        hx[i] = rnd(rnd_engine);
        hy[i] = rnd(rnd_engine);
        res += hx[i] * hy[i];
    }
    thrust::device_vector<float> dx = hx;
    thrust::device_vector<float> dy = hy;
    thrust::device_vector<float> dres(1, 0.0f);
    dot_wrapper CUDA_KERNEL(1, LENGTH)(LENGTH, dx.data().get(), dy.data().get(), dres.data().get());
    BOOST_REQUIRE_EQUAL(res, dres[0]);
}

__global__ void real_time_wrapper(const std::size_t n, const std::size_t nl, float* __restrict__ pt, const float* __restrict__ tlist)
{
    real_time(n, nl, pt, tlist);
}

BOOST_AUTO_TEST_CASE(real_time_test)
{
}
