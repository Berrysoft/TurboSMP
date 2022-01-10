#include <fsmp.cuh>

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
    const float mu_t, // LucyDDM 的估算 PE 数
    float* __restrict__ s0_history, // TRIALS, ||s||_0
    float* __restrict__ delta_nu_history, // TRIALS, Δν
    float* __restrict__ t0_history, // TRIALS, t_0
    float* __restrict__ A_vec,
    float* __restrict__ c_vec)
{
    // 波形编号，一个 block 一个波形
    const std::uint32_t waveform_id = blockIdx.x;
    // 并行用
    const std::uint32_t thread_id = threadIdx.x;
}

int main()
{
    return 0;
}
