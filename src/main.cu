#include <fsmp.cuh>

#include <curand.h>
#include <curand_kernel.h>

#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <numeric>

static constexpr std::size_t TRIALS = 2000;

__device__ int choose_step(float rnd)
{
    if (rnd < 0.25)
    {
        return -1;
    }
    else if (rnd < 0.5)
    {
        return 1;
    }
    else
    {
        return 2;
    }
}

__global__ void flow(
    const std::size_t nw,
    const std::size_t nl,
    float* __restrict__ cx, // nw x nl, Cov^-1 * A, 详见 FBMP
    const float* __restrict__ tlist, // nl
    float* __restrict__ z, // nw, residue waveform
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

    assert(blockDim.x == 1024);
    assert(nw * nl < 1024);
    assert(TRIALS > blockDim.x && TRIALS < blockDim.x * 2);

    curandState_t rnd_state;
    curand_init(id, 0, 0, &rnd_state);
    __shared__ float istar[TRIALS];
    __shared__ float home_s[TRIALS];
    istar[id] = curand_uniform(&rnd_state);
    home_s[id] = istar[id];
    if (id + blockDim.x < TRIALS)
    {
        istar[id + blockDim.x] = curand_uniform(&rnd_state);
        home_s[id + blockDim.x] = istar[id + blockDim.x];
    }

    __syncthreads();
    // TRIALS is a little bigger
    interp_by(TRIALS / 2, nl + 1, home_s, c_cha);
    interp_by(TRIALS / 2, nl + 1, &home_s[TRIALS / 2], c_cha);

    std::size_t NPE0 = (std::size_t)(mu_t + 0.5);
    float log_mu = std::log(mu_t);

    __shared__ float s[1024];
    if (id < NPE0)
    {
        s[id] = (id + 0.5) / (float)NPE0;
    }
    else
    {
        s[id] = 0;
    }
    assert(NPE0 < 1024);
    interp_by(NPE0, nl + 1, s, c_cha);
    __syncthreads();

    __shared__ float A_vec[1024];
    A_vec[id] = 0;
    __shared__ float c_vec[1024];
    c_vec[id] = 0;
    __shared__ float delta_nu;
    __shared__ float delta_cx[1024];
    delta_cx[id] = 0;
    __shared__ float delta_z[1024];
    delta_z[id] = 0;
    for (std::size_t i = 0; i < NPE0; i++)
    {
        combine(nw, nl, A, cx, s[i], A_vec, c_vec);
        __syncthreads();
        move(nw, nl, A_vec, c_vec, z, 1, mus, sig2s, A, &delta_nu, delta_cx, delta_z);
        if (id < nw * nl)
        {
            cx[id] += delta_cx[id];
        }
        if (id < nw)
        {
            z[id] += delta_z[id];
        }
        __syncthreads();
    }

    __shared__ float t0;
    if (id == 0)
    {
        if (NPE0 == 0)
        {
            t0 = tlist[0];
        }
        else
        {
            t0 = real_time(s[0], nl, tlist);
        }
    }

    __shared__ int flip[TRIALS];
    flip[id] = choose_step(curand_uniform(&rnd_state));
    if (id + blockDim.x < TRIALS)
    {
        flip[id + blockDim.x] = choose_step(curand_uniform(&rnd_state));
    }

    __shared__ float wanders[TRIALS];
    __shared__ float wts[TRIALS];
    __shared__ float accepts[TRIALS];
    __shared__ float accts[TRIALS];
    wanders[id] = curand_normal(&rnd_state);
    wts[id] = curand_normal(&rnd_state);
    accepts[id] = std::log(curand_uniform(&rnd_state));
    accts[id] = std::log(curand_uniform(&rnd_state));
    if (id + blockDim.x < TRIALS)
    {
        wanders[id + blockDim.x] = curand_normal(&rnd_state);
        wts[id + blockDim.x] = curand_normal(&rnd_state);
        accepts[id + blockDim.x] = std::log(curand_uniform(&rnd_state));
        accts[id + blockDim.x] = std::log(curand_uniform(&rnd_state));
    }

    __syncthreads();

    for (std::size_t i = 0; i < TRIALS; i++)
    {
        float t = istar[i];
        float step = flip[i];
        float home = home_s[i];
        float wander = wanders[i];
        float wt = wts[i];
        float accept = accepts[i];
        float acct = accts[i];

        // TODO
    }
}

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description program("TurboSMP");
    program.add_options()(
        "output,o", po::value<std::string>()->required(), "Output file")(
        "input,i", po::value<std::string>()->required(), "Input file")(
        "ref,r", po::value<std::string>()->required(), "Reference file");
    try
    {
        po::store(po::parse_command_line(argc, argv, program), vm);
        po::notify(vm);
    }
    catch (std::exception const& ex)
    {
        std::cerr << "Command line error: " << ex.what() << std::endl;
        std::cerr << program << std::endl;
        return 1;
    }

    auto input_name = vm["input"].as<std::string>();
    auto output_name = vm["output"].as<std::string>();
    auto ref_name = vm["ref"].as<std::string>();

    auto input = H5::H5File(input_name, H5F_ACC_RDONLY);
    auto ref = H5::H5File(ref_name, H5F_ACC_RDONLY);
    auto output = H5::H5File(output_name, H5F_ACC_TRUNC);

    // TODO: read files
    // TODO: call kernel
    // TODO: write output

    return 0;
}
