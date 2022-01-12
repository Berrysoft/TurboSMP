#include <fsmp.cuh>

#include <curand.h>
#include <curand_kernel.h>

#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <numeric>

static constexpr std::size_t TRIALS = 2000;

enum fsmp_step : int
{
    annihilate = -1,
    none = 0,
    generate = 1,
    move_one = 2,
};

__device__ __host__ fsmp_step choose_step(float rnd)
{
    if (rnd < 0.25)
    {
        return annihilate;
    }
    else if (rnd < 0.5)
    {
        return generate;
    }
    else
    {
        return move_one;
    }
}

__device__ __host__ void* memmove_back(void* dest, const void* src, size_t count)
{
    assert(dest < src);
    char* tmp = reinterpret_cast<char*>(dest);
    const char* s = reinterpret_cast<const char*>(src);
    while (count--)
    {
        *tmp++ = *s++;
    }
    return dest;
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
    float* __restrict__ t0_history, // TRIALS, t_0
    // Temp buffers
    float* __restrict__ istar, // TRIALS
    float* __restrict__ home_s, // TRIALS
    float* __restrict__ A_vec, // nw
    float* __restrict__ c_vec, // nw
    float* __restrict__ delta_cx, // nw x nl
    float* __restrict__ delta_z, // nw
    float* __restrict__ temp_cx, // nw x nl
    float* __restrict__ temp_z, // nw
    float* __restrict__ wanders, // TRIALS
    float* __restrict__ wts, // TRIALS
    float* __restrict__ accepts, // TRIALS
    float* __restrict__ accts, // TRIALS
    fsmp_step* __restrict__ flip // TRIALS
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
    std::size_t len_s = NPE0;
    interp_by(NPE0, nl + 1, s, c_cha);
    __syncthreads();

    __shared__ float delta_nu;
    __shared__ float beta;
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

    flip[id] = choose_step(curand_uniform(&rnd_state));
    if (id + blockDim.x < TRIALS)
    {
        flip[id + blockDim.x] = choose_step(curand_uniform(&rnd_state));
    }

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

    __shared__ float p1[1024];
    __shared__ float new_p1[1024];
    __shared__ float temp_p1[1024];
    for (std::size_t i = 0; i < TRIALS; i++)
    {
        __syncthreads();

        float t = istar[i];
        fsmp_step step = flip[i];
        float home = home_s[i];
        float wander = wanders[i];
        float wt = wts[i];
        float accept = accepts[i];
        float acct = accts[i];

        float new_t0 = t0 + wt;

        if (id < len_s)
        {
            p1[id] = s[id];
        }
        real_time(len_s, nl, p1, tlist);
        if (id < len_s)
        {
            new_p1[id] = p1[id] - new_t0;
            p1[id] -= t0;
        }
        lc(len_s, p1);
        lc(len_s, new_p1);
        if (id < len_s)
        {
            temp_p1[id] = new_p1[id] - p1[id];
        }
        sum(len_s, temp_p1);
        if (temp_p1[0] >= acct)
        {
            if (id == 0)
            {
                t0 = new_t0;
            }
            if (id < len_s)
            {
                p1[id] = new_p1[id];
            }
        }
        if (id == 0)
        {
            t0_history[i] = t0;
        }

        if (len_s == 0)
        {
            step = generate;
            accept += logf(4);
        }
        else if (len_s == 1 && step == -1)
        {
            accept -= logf(4);
        }

        if (id < nw)
        {
            A_vec[id] = 0;
            c_vec[id] = 0;
        }
        __syncthreads();

        if (step == generate)
        {
            if (home >= 0.5 && home <= nl - 0.5)
            {
                combine(nw, nl, A, cx, home, A_vec, c_vec);
                __syncthreads();
                move1(nw, nl, A_vec, c_vec, z, step, mus, sig2s, &delta_nu, &beta);
                if (id == 0)
                {
                    delta_nu += log_mu + lc(real_time(home, nl, tlist) - t0) - logf(p_cha[(std::size_t)home]) - logf(len_s + 1);
                }
                __syncthreads();
                if (delta_nu >= accept)
                {
                    move2(nw, nl, A_vec, c_vec, step, mus, A, beta, delta_cx, delta_z);
                    if (id == 0)
                    {
                        s[len_s] = home;
                        len_s++;
                    }
                }
            }
            else if (id == 0)
            {
                delta_nu = -NAN;
            }
        }
        else
        {
            std::size_t op = (std::size_t)(t * len_s);
            float loc = s[op];
            combine(nw, nl, A, cx, loc, A_vec, c_vec);
            __syncthreads();
            move1(nw, nl, A_vec, c_vec, z, annihilate, mus, sig2s, &delta_nu, &beta);
            __syncthreads();
            if (step == annihilate)
            {
                if (id == 0)
                {
                    delta_nu -= log_mu + p1[op] - logf(p_cha[(std::size_t)loc]) - logf(len_s);
                }
                __syncthreads();
                if (delta_nu >= accept)
                {
                    move2(nw, nl, A_vec, c_vec, annihilate, mus, A, beta, delta_cx, delta_z);
                    if (id == 0)
                    {
                        memmove_back(&s[op], &s[op + 1], len_s - op - 1);
                        len_s--;
                    }
                }
            }
            else if (step == move_one)
            {
                float nloc = loc + wander;
                if (nloc >= 0.5 && nloc <= nl - 0.5)
                {
                    move2(nw, nl, A_vec, c_vec, annihilate, mus, A, beta, delta_cx, delta_z);
                    if (id < nw * nl)
                    {
                        temp_cx[id] = cx[id] + delta_cx[id];
                    }
                    if (id < nw)
                    {
                        temp_z[id] = z[id] + delta_z[id];
                    }
                    if (id < nw)
                    {
                        A_vec[id] = 0;
                        c_vec[id] = 0;
                    }
                    __syncthreads();
                    combine(nw, nl, A, temp_cx, nloc, A_vec, c_vec);
                    __syncthreads();
                    __shared__ float delta_nu1;
                    move1(nw, nl, A_vec, c_vec, temp_z, generate, mus, sig2s, &delta_nu1, &beta);
                    if (id == 0)
                    {
                        delta_nu += delta_nu1 + lc(real_time(nloc, nl, tlist) - t0) - p1[op];
                    }
                    __syncthreads();
                    if (delta_nu >= accept)
                    {
                        move2(nw, nl, A_vec, c_vec, generate, mus, A, beta, temp_cx, temp_z);
                        if (id == 0)
                        {
                            s[op] = nloc;
                        }
                        if (id < nw * nl)
                        {
                            delta_cx[id] += temp_cx[id];
                        }
                        if (id < nw)
                        {
                            delta_z[id] += temp_z[id];
                        }
                    }
                }
                else
                {
                    if (id == 0)
                    {
                        delta_nu = -NAN;
                    }
                }
            }
        }

        __syncthreads();

        if (delta_nu >= accept)
        {
            if (id < nw * nl)
            {
                cx[id] += delta_cx[id];
            }
            if (id < nw)
            {
                z[id] += delta_z[id];
            }
        }
        else
        {
            if (id == 0)
            {
                delta_nu = 0;
                step = none;
            }
        }

        if (id == 0)
        {
            delta_nu_history[i] = delta_nu;
            flip[i] = step;
        }
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
