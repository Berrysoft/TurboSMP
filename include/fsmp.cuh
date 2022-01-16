#pragma once

#ifdef __INTELLISENSE__
    #define CUDA_KERNEL(...)
    #define __CUDACC__
#else
    #define CUDA_KERNEL(...) <<< __VA_ARGS__ >>>
#endif

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cuda_runtime.h>

template <typename T>
__device__ __host__ constexpr T div_ceil(T a, T b)
{
    return a / b + (a % b ? 1 : 0);
}

__device__ void sum(const std::size_t n, float* x)
{
    if (n <= 1) return;
    const std::uint32_t id = threadIdx.x;
    if (n % 2 && id == 0)
    {
        x[0] += x[n - 1];
    }
    std::size_t num = n;
    while (num /= 2)
    {
        std::uint32_t nslice = div_ceil<std::uint32_t>(num, blockDim.x);
        for (std::uint32_t i = 0; i < nslice; i++)
        {
            std::uint32_t index = id + i * blockDim.x;
            if (index < num)
            {
                x[index] += x[index + num];
            }
        }
        if (id == 0 && num % 2 == 1)
        {
            x[num] = 0;
        }
        __syncthreads();
        if (num == 1)
        {
            break;
        }
        else if (num % 2 == 1)
        {
            num++;
        }
    }
}

__device__ void sum2x(const std::size_t nx, const std::size_t ny, float* x)
{
    if (nx <= 1) return;
    assert(ny <= blockDim.x);
    const std::uint32_t id = threadIdx.x;
    const std::uint32_t ix = id / ny;
    const std::uint32_t dimx = div_ceil<std::uint32_t>(blockDim.x, ny);
    const std::uint32_t iy = id % ny;
    if (nx % 2 && ix == 0)
    {
        x[iy] += x[(nx - 1) * ny + iy];
    }
    std::size_t num = nx;
    while (num /= 2)
    {
        const std::uint32_t nslice = div_ceil<std::uint32_t>(num, dimx);
        for (std::uint32_t i = 0; i < nslice; i++)
        {
            std::uint32_t index = ix + i * dimx;
            if (index < num)
            {
                x[index * ny + iy] += x[(index + num) * ny + iy];
            }
        }
        if (ix == 0 && num % 2 == 1)
        {
            x[num * ny + iy] = 0;
        }
        __syncthreads();
        if (num == 1)
        {
            break;
        }
        else if (num % 2 == 1)
        {
            num++;
        }
    }
}

// Thread 0 can assume the result is calculated successfully.
__device__ void dot(const std::size_t n, const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ pres)
{
    assert(n <= 1024);

    __shared__ float temp[1024];
    const std::uint32_t id = threadIdx.x;
    const std::uint32_t nslice = div_ceil<std::uint32_t>(n, blockDim.x);
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < n)
        {
            temp[index] = x[index] * y[index];
        }
    }
    __syncthreads();
    sum(n, temp);
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
    float frac = std::modf(t - 0.5, &fti);
    std::uint32_t ti = (std::uint32_t)fti;

    const std::uint32_t id = threadIdx.x;
    const std::uint32_t nslice = div_ceil<std::uint32_t>(nw, blockDim.x);
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < nw)
        {
            A_vec[index] = (1 - frac) * A[index * nl + ti] + frac * A[index * nl + ti + 1];
            c_vec[index] = (1 - frac) * cx[index * nl + ti] + frac * cx[index * nl + ti + 1];
        }
    }
}

__device__ __host__ std::size_t bisect(const std::size_t n, const float* arr, const float x)
{
    std::size_t off = 0;
    std::size_t num = n;
    std::size_t i;
    do
    {
        i = off + num / 2;
        if (arr[i] == x)
        {
            return i;
        }
        else if (arr[i] < x)
        {
            off = i + 1;
            num = n - off;
        }
        else
        {
            num = i - off;
        }
    } while (num);
    return i;
}

__device__ __host__ float interp_id(const float t, const std::size_t nf, const float* f)
{
    std::size_t i = bisect(nf, f, t);
    if (i == 0)
    {
        // assert(t == f[0]);
        return 0;
    }
    else
    {
        assert(i < nf);
        return i - 1 + (t - f[i - 1]) / (f[i] - f[i - 1]);
    }
}

__device__ void interp_id(const std::size_t nx, const std::size_t nf, float* __restrict__ x, const float* __restrict__ f)
{
    const std::uint32_t id = threadIdx.x;
    const std::uint32_t nslice = div_ceil<std::uint32_t>(nx, blockDim.x);
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < nx)
        {
            x[index] = interp_id(x[index], nf, f);
        }
    }
}

__device__ __host__ float interp_by(const float t, const std::size_t nf, const float* by)
{
    std::size_t i = (std::size_t)std::ceil(t);
    if (i == 0)
    {
        assert(t == 0);
        return by[0];
    }
    else
    {
        assert(i < nf);
        return by[i - 1] + (by[i] - by[i - 1]) * (t - (i - 1));
    }
}

__device__ void interp_by(const std::size_t nx, const std::size_t nf, float* __restrict__ x, const float* __restrict__ by)
{
    const std::uint32_t id = threadIdx.x;
    const std::uint32_t nslice = div_ceil<std::uint32_t>(nx, blockDim.x);
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < nx)
        {
            x[index] = interp_by(x[index], nf, by);
        }
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
    interp_id(n, nl, pt, tlist);
}

__device__ float real_time(float t, const std::size_t nl, const float* tlist)
{
    return real_time(t, nl, tlist);
}

// log of the light curve, t is t0-subtracted.
// the shape is exGaussian, with paramaters tau=20ns
// sigma=5ns.
__device__ __host__ float lc(const float t)
{
    constexpr float tau = 20.;
    constexpr float alpha = 1. / tau;
    constexpr float sigma = 5.;

    float co = -std::log(2.0 * tau) + alpha * alpha * sigma * sigma / 2.0;

    float x_erf = (alpha * sigma * sigma - t) / (std::sqrt(2.0) * sigma);
    return co + std::log(1.0 - std::erf(x_erf)) - alpha * t;
}

__device__ void lc(const std::size_t n, float* t)
{
    const std::uint32_t id = threadIdx.x;
    const std::uint32_t nslice = div_ceil<std::uint32_t>(n, blockDim.x);
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < n)
        {
            t[index] = lc(t[index]);
        }
    }
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
    // Sync because the shared memory may be overwritten
    __syncthreads();
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

__device__ void sum_minus_n_m_mp(
    const std::size_t nw,
    const std::size_t nl,
    const float beta,
    const float* __restrict__ c_vec, // nw
    const float* __restrict__ A, // nw x nl
    float* __restrict__ delta_cx // nw x nl
)
{
    const std::uint32_t id = threadIdx.x;
    const std::size_t n2 = nw * nl;
    const std::uint32_t nslice = div_ceil<std::uint32_t>(n2, blockDim.x);
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < n2)
        {
            delta_cx[index] = A[index] * c_vec[index / nl];
        }
    }
    __syncthreads();
    sum2x(nw, nl, delta_cx);
    __shared__ float temp[1024];
    const std::uint32_t nl_nslice = div_ceil<std::uint32_t>(nl, blockDim.x);
    for (std::uint32_t i = 0; i < nl_nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < nl)
        {
            temp[index] = delta_cx[index];
        }
    }
    __syncthreads();
    for (std::uint32_t i = 0; i < nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < n2)
        {
            delta_cx[index] = -beta * c_vec[index / nl] * temp[index % nl];
        }
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
    const std::uint32_t id = threadIdx.x;
    sum_minus_n_m_mp(nw, nl, beta, c_vec, A, delta_cx);
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
