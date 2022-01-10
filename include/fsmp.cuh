#pragma once

#ifdef __INTELLISENSE__
    #define CUDA_KERNEL(...)
    #define __CUDACC__
#else
    #define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#include <cassert>
#include <cstdint>
#include <cuda_runtime.h>

__device__ void sum(const std::size_t n, float* x)
{
    assert(n % 2 == 0);
    const std::uint32_t id = threadIdx.x;
    std::size_t num = n / 2;
    while (num)
    {
        if (id < num)
        {
            x[id] += x[id + num];
        }
        else
        {
            break;
        }
        __syncthreads();
        num /= 2;
    }
}

// Thread 0 can assume the result is calculated successfully.
__device__ void dot(const std::size_t n, const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ pres)
{
    __shared__ float temp[1024];
    assert(n <= 1024);
    const std::uint32_t id = threadIdx.x;
    if (id < n)
    {
        temp[id] = x[id] * y[id];
    }
    else if (id < 1024)
    {
        // Set to zero because temp is reused.
        temp[id] = 0;
    }
    __syncthreads();
    sum(1024, temp);
    if (id == 0) *pres = temp[0];
}

__device__ void combine(
    const std::size_t nw,
    const std::size_t nl,
    const float* __restrict__ A,
    const float* __restrict__ cx,
    const float t, // 时间，属于 tlist 下标
    float* __restrict__ A_vec, // nw
    float* __restrict__ c_vec // nw
)
{
    float fti;
    float frac = modf(t - 0.5, &fti);
    std::uint32_t ti = (std::uint32_t)fti;

    float alpha[2] = { 1 - frac, frac };

    const std::uint32_t id = threadIdx.x;
    assert(id < nw * 4);

    if (id < nw)
    {
        A_vec[id] = 0;
        c_vec[id] = 0;
    }
    // Sync because nw < nw * 4
    __syncthreads();
    if (id < nw * 4)
    {
        auto id_vec = id / 4;
        auto id_dot = id % 4;
        auto id_alpha = id_dot / 2;
        auto id_interp = id_dot % 2;
        atomicAdd(&A_vec[id_vec], alpha[id_alpha] * A[id_vec * nl + id_interp + ti]);
        atomicAdd(&c_vec[id_vec], alpha[id_alpha] * cx[id_vec * nl + id_interp + ti]);
    }
}

// TODO: t is ordered
__device__ void real_time(
    const std::size_t n,
    const std::size_t nl,
    float* __restrict__ pt, // n
    const float* __restrict__ tlist // nl
)
{
    const std::uint32_t id = threadIdx.x;
    if (id < n)
    {
        float t = pt[id];
        // TODO: bisect
        std::size_t i = 0;
        for (; i < nl; i++)
        {
            if (tlist[i] >= t) break;
        }
        if (i == 0)
        {
            assert(t == tlist[0]);
            pt[id] = 0;
        }
        else
        {
            assert(i < nl);
            pt[id] = i - 1 + (t - tlist[i - 1]) / (tlist[i] - tlist[i - 1]);
        }
    }
}

__device__ void lc(const std::size_t n, float* t)
{
}

__device__ void move1(
    const std::size_t nw,
    const std::size_t nl,
    const float* __restrict__ A_vec,
    const float* __restrict__ c_vec,
    const float* __restrict__ z,
    const int step,
    const float mus,
    const float sig2s,
    float* __restrict__ pdelta_nu, // x1
    float* __restrict__ pbeta // x1
)
{
    const std::uint32_t id = threadIdx.x;
    float beta_under;
    dot(nw, A_vec, c_vec, &beta_under);
    float temp;
    dot(nw, z, c_vec, &temp);
    if (id == 0)
    {
        float fsig2s = step * sig2s;
        beta_under = 1 + fsig2s * beta_under;
        float beta = fsig2s / beta_under;
        float delta_nu = 0.5 * (beta * powf(temp + mus / sig2s, 2.0f) - powf(mus, 2.0f) / sig2s);
        delta_nu -= 0.5 * logf(beta_under);
        *pdelta_nu = delta_nu;
        *pbeta = beta;
    }
}

__device__ void move2(
    const std::size_t nw,
    const std::size_t nl,
    const float* __restrict__ A_vec,
    const float* __restrict__ c_vec,
    const int step,
    const float mus,
    const float* __restrict__ A,
    const float beta,
    float* __restrict__ delta_cx, // nw x nl
    float* __restrict__ delta_z // nw
)
{
    assert(nw * nl < 1024);
    const std::uint32_t id = threadIdx.x;
    if (id < nw * nl)
    {
        delta_cx[id] = A[id] * c_vec[id / nl];
    }
    __syncthreads();
    __shared__ float temp[1024];
    assert(nw < 1024);
    if (id < nl)
    {
        // TODO: optimize sum
        float res = 0.0;
        for (std::size_t i = 0; i < nw; i++)
        {
            res += delta_cx[i * nl + id];
        }
        temp[id] = res;
    }
    __syncthreads();
    if (id < nw * nl)
    {
        delta_cx[id] = beta * c_vec[id / nl] * temp[id % nl];
    }
    if (id < nw)
    {
        delta_z[id] = -step * A_vec[id] * mus;
    }
}

// return delta_nu
__device__ void move(
    const std::size_t nw,
    const std::size_t nl,
    const float* __restrict__ A_vec,
    const float* __restrict__ c_vec,
    const float* __restrict__ z,
    const int step,
    const float mus,
    const float sig2s,
    const float* A,
    float* __restrict__ delta_nu,
    float* __restrict__ delta_cx,
    float* __restrict__ delta_z)
{
    __shared__ float beta;
    move1(nw, nl, A_vec, c_vec, z, step, mus, sig2s, delta_nu, &beta);
    // beta is used after __syncthreads()
    // __syncthreads();
    move2(nw, nl, A_vec, c_vec, step, mus, A, beta, delta_cx, delta_z);
}
