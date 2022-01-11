#include <fsmp.cuh>

#include <curand.h>
#include <curand_kernel.h>
#include <numeric>

static constexpr std::size_t TRIALS = 2000;

__global__ void flow(
    const std::size_t nw,
    const std::size_t nl,
    const float* __restrict__ cx, // nw x nl, Cov^-1 * A, 详见 FBMP
    const float* __restrict__ tlist, // nl
    const float* __restrict__ z, // nw, residue waveform
    const float mus, // spe波形的平均幅值
    const float sig2s, // spe波形幅值方差
    const float* __restrict__ A, // nw x nl
    const float* __restrict__ p_cha, // nl
    const float* __restrict__ c_cha, // nl + 1, with 0 padding in front
    const float mu_t, // LucyDDM 的估算 PE 数
    float* __restrict__ s0_history, // TRIALS, ||s||_0
    float* __restrict__ delta_nu_history, // TRIALS, Δν
    float* __restrict__ t0_history // TRIALS, t_0
)
{
    // 波形编号，一个 block 一个波形
    const std::uint32_t waveform_id = blockIdx.x;
    // 并行用
    const std::uint32_t id = threadIdx.x;

    curandState_t rnd_state;
    curand_init(id, 0, 0, &rnd_state);
    __shared__ float home_s[TRIALS];
    assert(TRIALS > blockDim.x && TRIALS < blockDim.x * 2);
    home_s[id] = curand_uniform(&rnd_state);
    if (id * 2 < TRIALS)
    {
        home_s[id * 2] = curand_uniform(&rnd_state);
    }
    __syncthreads();
    // TRIALS is a little bigger
    interp_by(TRIALS / 2, nl + 1, home_s, c_cha);
    interp_by(TRIALS / 2, nl + 1, &home_s[TRIALS / 2], c_cha);

    std::size_t NPE0 = (std::size_t)(mu_t + 0.5);

    // TODO
}

int main()
{
    return 0;
}
