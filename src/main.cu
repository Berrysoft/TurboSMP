#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda_runtime.h>

#ifdef __INTELLISENSE__
    #define CUDA_KERNEL(...)
    #include <device_launch_parameters.h>
#else
    #define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

using namespace std;

__device__ float dot(const size_t n, const float* x, const float* y)
{
    const uint32_t id = threadIdx.x;
    float result = 0.0;
    if (id < n)
    {
        result += x[id] * y[id];
    }
    return result;
}

__device__ void combine(
    const size_t nw,
    const float* A,
    const float* cx,
    const float t, // 时间，属于 tlist 下标
    float* A_vec, // nw
    float* c_vec // nw
)
{
}

// TODO: t is ordered
// TODO: bisect
__device__ __host__ void real_time(
    const size_t n,
    const size_t nl,
    float* pt, // n
    const float* tlist // nl
)
{
    const uint32_t id = threadIdx.x;
    if (id < n)
    {
        float t = pt[id];
        size_t i = 0;
        for (; i < nl; i++)
        {
            if (tlist[i] > t) break;
        }
        assert(i > 0);
        assert(i < nl);
        pt[id] = i - 1 + (t - tlist[i - 1]) / (tlist[i] - tlist[i - 1]);
    }
}

__device__ __host__ void lc(const size_t n, float* t)
{
}

__device__ void move1(
    const size_t nw,
    const size_t nl,
    const float* A_vec,
    const float* c_vec,
    const float* z,
    const int step,
    const float mus,
    const float sig2s,
    float* delta_nu, // x1
    float* beta // x1
)
{
    float fsig2s = step * sig2s;
    float beta_under = 1 + fsig2s * dot(nw, A_vec, c_vec);
    // TODO
}

__device__ void move2(
    const size_t nw,
    const size_t nl,
    const float* A_vec,
    const float* c_vec,
    const int step,
    const float mus,
    const float* A,
    const float beta,
    float* delta_cx, // nw x nl
    float* delta_z // nw
)
{
}

// return delta_nu
__device__ float move(
    const size_t nw,
    const size_t nl,
    const float* A_vec,
    const float* c_vec,
    const float* z,
    const int step,
    const float mus,
    const float sig2s,
    const float* A,
    float* delta_cx,
    float* delta_z)
{
    float delta_nu, beta;
    move1(nw, nl, A_vec, c_vec, z, step, mus, sig2s, &delta_nu, &beta);
    move2(nw, nl, A_vec, c_vec, step, mus, A, beta, delta_cx, delta_z);
    return delta_nu;
}

static constexpr size_t TRIALS = 2000;

__global__ void flow(
    const size_t nw,
    const size_t nl,
    const float* cx, // nw x nl, Cov^-1 * A, 详见 FBMP
    const float* tlist, // nl
    const float* z, // nw, residue waveform
    const float mus, // spe波形的平均幅值
    const float sig2s, // spe波形幅值方差
    const float* A, // nw x nl
    const float* p_cha, // nl
    const float mu_t, // LucyDDM 的估算 PE 数
    float* s0_history, // TRIALS, ||s||_0
    float* delta_nu_history, // TRIALS, Δν
    float* t0_history, // TRIALS, t_0
    float* A_vec,
    float* c_vec)
{
    // 波形编号，一个 block 一个波形
    const uint32_t waveform_id = blockIdx.x;
    // 并行用
    const uint32_t thread_id = threadIdx.x;
}

int main()
{
    return 0;
}
