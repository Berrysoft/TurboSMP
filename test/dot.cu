#define BOOST_TEST_MODULE DotTest
#include <boost/test/included/unit_test.hpp>
#include <boost/test/tools/floating_point_comparison.hpp>

#include <fsmp.cuh>

#include <algorithm>
#include <numeric>
#include <random>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

std::random_device rnd_device{};
std::default_random_engine rnd_engine{ rnd_device() };

BOOST_AUTO_TEST_CASE(bisect_test)
{
    float arr[] = { 1, 2, 3, 4, 5, 6, 7 };
    BOOST_REQUIRE_EQUAL(0, bisect(6, arr, 1));
    BOOST_REQUIRE_EQUAL(1, bisect(6, arr, 1.5));
    BOOST_REQUIRE_EQUAL(1, bisect(6, arr, 2));
    BOOST_REQUIRE_EQUAL(5, bisect(6, arr, 5.5));
    BOOST_REQUIRE_EQUAL(5, bisect(6, arr, 6));

    BOOST_REQUIRE_EQUAL(0, bisect(7, arr, 1));
    BOOST_REQUIRE_EQUAL(1, bisect(7, arr, 1.5));
    BOOST_REQUIRE_EQUAL(1, bisect(7, arr, 2));
    BOOST_REQUIRE_EQUAL(5, bisect(7, arr, 5.5));
    BOOST_REQUIRE_EQUAL(6, bisect(7, arr, 7));
}

__global__ void sum_wrapper(const std::size_t n, float* x)
{
    sum(n, x);
}

BOOST_AUTO_TEST_CASE(sum_test)
{
    constexpr size_t N = 101;
    thrust::host_vector<float> hx(N, 0.0f);
    std::uniform_real_distribution<float> rnd{};
    for (size_t i = 0; i < N; i++)
    {
        hx[i] = rnd(rnd_engine);
    }

    float expect_sum = std::accumulate(hx.begin(), hx.end(), 0.0f);

    thrust::device_vector<float> dx = hx;
    sum_wrapper CUDA_KERNEL(1, 9)(N, dx.data().get());

    float rsum = dx[0];
    BOOST_REQUIRE_CLOSE(expect_sum, rsum, 1e-3f);
}

__global__ void sum2x_wrapper(const std::size_t nx, const std::size_t ny, float* x)
{
    sum2x(nx, ny, x);
}

BOOST_AUTO_TEST_CASE(sum2x_test)
{
    constexpr size_t NX = 101;
    constexpr size_t NY = 11;
    thrust::host_vector<float> hx(NX * NY, 0.0f);
    std::uniform_real_distribution<float> rnd{};
    for (size_t i = 0; i < NX * NY; i++)
    {
        hx[i] = rnd(rnd_engine);
    }

    thrust::host_vector<float> expect_sum(NY, 0.0f);
    for (size_t j = 0; j < NY; j++)
    {
        float sum = 0.0;
        for (size_t i = 0; i < NX; i++)
        {
            sum += hx[i * NY + j];
        }
        expect_sum[j] = sum;
    }

    thrust::device_vector<float> dx = hx;
    sum2x_wrapper CUDA_KERNEL(1, 1024)(NX, NY, dx.data().get());

    for (size_t j = 0; j < NY; j++)
    {
        BOOST_REQUIRE_CLOSE(expect_sum[j], (float)dx[j], 1e-3f);
    }
}

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
    real_time_wrapper CUDA_KERNEL(1, 5)(dts.size(), dtlist.size(), dts.data().get(), dtlist.data().get());

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
    real_time_wrapper CUDA_KERNEL(1, 10)(NT, NTLIST, dts.data().get(), dtlist.data().get());

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

__global__ void interp_by_wrapper(const std::size_t nx, const std::size_t nf, float* __restrict__ x, const float* __restrict__ by)
{
    interp_by(nx, nf, x, by);
}

BOOST_AUTO_TEST_CASE(interp_by_test)
{
    std::vector<float> hts{ 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    std::vector<float> htlist{ 100, 102, 104, 106, 108, 110 };
    std::vector<float> expect{ 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110 };

    thrust::device_vector<float> dts = hts;
    thrust::device_vector<float> dtlist = htlist;
    interp_by_wrapper CUDA_KERNEL(1, 5)(dts.size(), dtlist.size(), dts.data().get(), dtlist.data().get());

    BOOST_REQUIRE_EQUAL_COLLECTIONS(expect.begin(), expect.end(), dts.begin(), dts.end());
}

__global__ void lc_wrapper(const std::size_t n, float* t)
{
    lc(n, t);
}

BOOST_AUTO_TEST_CASE(lc_test)
{
    std::vector<float> real_time{ 0, 1, 2 };
    std::vector<float> log_lc{ -3.87754404, -3.74832397, -3.64498369 };
    thrust::device_vector<float> drt = real_time;
    lc_wrapper CUDA_KERNEL(1, 2)(drt.size(), drt.data().get());

    for (size_t j = 0; j < real_time.size(); j++)
    {
        BOOST_REQUIRE_CLOSE(log_lc[j], (float)drt[j], 1e-3f);
    }
}

__global__ void sum_minus_n_m_mp_wrapper(
    const std::size_t nw,
    const std::size_t nl,
    const float beta,
    const float* __restrict__ c_vec,
    const float* __restrict__ A,
    float* __restrict__ delta_cx)
{
    sum_minus_n_m_mp(nw, nl, beta, c_vec, A, delta_cx);
}

BOOST_AUTO_TEST_CASE(einsum_test)
{
    constexpr size_t nw = 4;
    constexpr size_t nl = 3;
    const float beta = 1.0;
    std::vector<float> c_vec = { 1, 2, 3, 4 };
    std::vector<float> A = { 1, 2, 3,
                             2, 2, 3,
                             3, 2, 3,
                             4, 2, 3 };

    thrust::device_vector<float> dc_vec = c_vec;
    thrust::device_vector<float> dA = A;
    thrust::device_vector<float> delta_cx(nw * nl, 0.0f);
    sum_minus_n_m_mp_wrapper CUDA_KERNEL(1, 1024)(nw, nl, beta, dc_vec.data().get(), dA.data().get(), delta_cx.data().get());

    std::vector<float> expect = { -30, -20, -30,
                                  -60, -40, -60,
                                  -90, -60, -90,
                                  -120, -80, -120 };

    BOOST_REQUIRE_EQUAL_COLLECTIONS(expect.begin(), expect.end(), delta_cx.begin(), delta_cx.end());
}
