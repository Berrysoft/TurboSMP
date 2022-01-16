#include <fsmp.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <curand.h>
#include <curand_kernel.h>

#include <H5Cpp.h>
#include <boost/program_options.hpp>
#include <iostream>
#include <numeric>
#include <vector>

static constexpr std::uint32_t TRIALS = 2000;

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

__device__ void flow_one(
    const std::uint32_t nw,
    const std::uint32_t nl,
    float* __restrict__ cx, // nw x nl, Cov^-1 * A, 详见 FBMP
    const float* __restrict__ tlist, // nl
    float* __restrict__ z, // nw, residue waveform
    const float mus, // spe波形的平均幅值
    const float sig2s, // spe波形幅值方差
    const float sig2w,
    const float* __restrict__ A, // nw x nl
    const float* __restrict__ p_cha, // nl
    const float* __restrict__ c_cha, // nl + 1, with 0 padding in front
    const float mu_t, // LucyDDM 的估算 PE 数
    std::uint32_t* __restrict__ s0_history, // TRIALS, ||s||_0
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
    const std::uint32_t id = threadIdx.x;

    curandState_t rnd_state;
    curand_init(id, 0, 0, &rnd_state);

    const std::uint32_t trials_nslice = div_ceil(TRIALS, blockDim.x);
    const std::uint32_t n2 = nw * nl;
    const std::uint32_t n2_nslice = div_ceil(n2, blockDim.x);

    for (std::uint32_t i = 0; i < trials_nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < TRIALS)
        {
            istar[index] = curand_uniform(&rnd_state);
            home_s[index] = istar[index];
        }
    }

    // cx is initialized as A
    for (std::uint32_t i = 0; i < n2_nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < n2)
        {
            cx[index] /= sig2w;
        }
    }

    __syncthreads();
    interp_by(TRIALS, nl + 1, home_s, c_cha);

    std::uint32_t NPE0 = (std::uint32_t)(mu_t + 0.5);
    float log_mu = std::log(mu_t);

    __shared__ float s[1024];
    if (id < NPE0)
    {
        s[id] = (id + 0.5) / (float)NPE0;
    }
    assert(NPE0 < 1024);
    std::uint32_t len_s = NPE0;
    interp_by(len_s, nl + 1, s, c_cha);
    __syncthreads();

    __shared__ float delta_nu;
    __shared__ float beta;
    for (std::uint32_t i = 0; i < NPE0; i++)
    {
        combine(nw, nl, A, cx, s[i], A_vec, c_vec);
        __syncthreads();
        move(nw, nl, A_vec, c_vec, z, 1, mus, sig2s, A, &delta_nu, delta_cx, delta_z);
        for (std::uint32_t j = 0; j < n2_nslice; j++)
        {
            std::uint32_t jndex = id + j * blockDim.x;
            if (jndex < n2)
            {
                cx[jndex] += delta_cx[jndex];
            }
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

    for (std::uint32_t i = 0; i < trials_nslice; i++)
    {
        std::uint32_t index = id + i * blockDim.x;
        if (index < TRIALS)
        {
            flip[index] = choose_step(curand_uniform(&rnd_state));
            wanders[index] = curand_normal(&rnd_state);
            wts[index] = curand_normal(&rnd_state);
            accepts[index] = std::log(curand_uniform(&rnd_state));
            accts[index] = std::log(curand_uniform(&rnd_state));
        }
    }

    __shared__ float p1[1024];
    __shared__ float new_p1[1024];
    __shared__ float temp_p1[1024];
    for (std::uint32_t i = 0; i < TRIALS; i++)
    {
        assert(len_s <= blockDim.x);
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
                    delta_nu += log_mu + lc(real_time(home, nl, tlist) - t0) - logf(p_cha[(std::uint32_t)home]) - logf(len_s + 1);
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
            std::uint32_t op = (std::uint32_t)(t * len_s);
            float loc = s[op];
            combine(nw, nl, A, cx, loc, A_vec, c_vec);
            __syncthreads();
            move1(nw, nl, A_vec, c_vec, z, annihilate, mus, sig2s, &delta_nu, &beta);
            __syncthreads();
            if (step == annihilate)
            {
                if (id == 0)
                {
                    delta_nu -= log_mu + p1[op] - logf(p_cha[(std::uint32_t)loc]) - logf(len_s);
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
                    for (std::uint32_t j = 0; j < n2_nslice; j++)
                    {
                        std::uint32_t jndex = id + j * blockDim.x;
                        if (jndex < n2)
                        {
                            temp_cx[jndex] = cx[jndex] + delta_cx[jndex];
                        }
                    }
                    if (id < nw)
                    {
                        temp_z[id] = z[id] + delta_z[id];
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
                        for (std::uint32_t j = 0; j < n2_nslice; j++)
                        {
                            std::uint32_t jndex = id + j * blockDim.x;
                            if (jndex < n2)
                            {
                                delta_cx[jndex] += temp_cx[jndex];
                            }
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
            for (std::uint32_t j = 0; j < n2_nslice; j++)
            {
                std::uint32_t jndex = id + j * blockDim.x;
                if (jndex < n2)
                {
                    cx[jndex] += delta_cx[jndex];
                }
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
            s0_history[i] = len_s;
            delta_nu_history[i] = delta_nu;
            flip[i] = step;
        }
    }
}

__global__ void flow(
    const std::size_t NW,
    const std::size_t mnw, // max nw
    const std::size_t mnl, // max nl
    const std::uint32_t* pnw, // NW
    const std::uint32_t* pnl, // NW
    float* __restrict__ cx, // NW x mnw x mnl
    const float* __restrict__ tlist, // NW x mnl
    float* __restrict__ z, // NW x mnw
    const float* __restrict__ mus,
    const float* __restrict__ sig2s,
    const float* __restrict__ sig2w,
    const float* __restrict__ A, // NW x mnw x mnl
    const float* __restrict__ p_cha, // NW x mnl
    const float* __restrict__ c_cha, // NW x (mnl + 1)
    const float* __restrict__ mu_t, // NW
    std::uint32_t* __restrict__ s0_history, // NW x TRIALS
    float* __restrict__ delta_nu_history, // NW x TRIALS
    float* __restrict__ t0_history, // NW x TRIALS
    // Temp buffers
    float* __restrict__ istar, // NW x TRIALS
    float* __restrict__ home_s, // NW x TRIALS
    float* __restrict__ A_vec, // NW x mnw
    float* __restrict__ c_vec, // NW x mnw
    float* __restrict__ delta_cx, // NW x mnw x mnl
    float* __restrict__ delta_z, // NW x mnw
    float* __restrict__ temp_cx, // NW x mnw x mnl
    float* __restrict__ temp_z, // NW x mnw
    float* __restrict__ wanders, // NW x TRIALS
    float* __restrict__ wts, // NW x TRIALS
    float* __restrict__ accepts, // NW x TRIALS
    float* __restrict__ accts, // NW x TRIALS
    fsmp_step* __restrict__ flip // NW x TRIALS
)
{
    const std::uint32_t wid = blockIdx.x;
    const std::size_t mn2 = mnw * mnl;

    if (wid < NW)
    {
        flow_one(
            pnw[wid],
            pnl[wid],
            cx + wid * mn2,
            tlist + wid * mnl,
            z + wid * mnw,
            mus[wid],
            sig2s[wid],
            sig2w[wid],
            A + wid * mn2,
            p_cha + wid * mnl,
            c_cha + wid * (mnl + 1),
            mu_t[wid],
            s0_history + wid * TRIALS,
            delta_nu_history + wid * TRIALS,
            t0_history + wid * TRIALS,
            istar + wid * TRIALS,
            home_s + wid * TRIALS,
            A_vec + wid * mnw,
            c_vec + wid * mnw,
            delta_cx + wid * mn2,
            delta_z + wid * mnw,
            temp_cx + wid * mn2,
            temp_z + wid * mnw,
            wanders + wid * TRIALS,
            wts + wid * TRIALS,
            accepts + wid * TRIALS,
            accts + wid * TRIALS,
            flip + wid * TRIALS);
    }
}

int main(int argc, char** argv)
{
    namespace po = boost::program_options;
    po::variables_map vm;
    po::options_description program("TurboSMP");
    program.add_options()(
        "output,o", po::value<std::string>()->required(), "Output file")(
        "input,i", po::value<std::string>()->required(), "Input file");
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

    auto input = H5::H5File(input_name, H5F_ACC_RDONLY);

    auto float_type = H5::PredType::NATIVE_FLOAT;

    auto A_data = input.openDataSet("A");
    auto A_space = A_data.getSpace();
    hsize_t A_size[3];
    A_space.getSimpleExtentDims(A_size);
    std::size_t NW = A_size[0];
    std::size_t mnw = A_size[1];
    std::size_t mnl = A_size[2];

    std::vector<float> A(NW * mnw * mnl, 0.0f);
    A_data.read(A.data(), float_type);

    std::vector<float> z(NW * mnw, 0.0f);
    input.openDataSet("z").read(z.data(), float_type);

    auto tq_data = input.openDataSet("tq");

    std::vector<float> tlist(NW * mnl, 0.0f);
    std::vector<float> p_cha(NW * mnl, 0.0f);
    {
        H5::CompType t(sizeof(float));
        t.insertMember("t_s", 0, float_type);
        tq_data.read(tlist.data(), t);
    }
    {
        H5::CompType t(sizeof(float));
        t.insertMember("q_s", 0, float_type);
        tq_data.read(p_cha.data(), t);
    }

    std::vector<float> c_cha(NW * (mnl + 1), 0.0f);
    for (std::size_t i = 0; i < NW; i++)
    {
        std::partial_sum(p_cha.begin() + i * mnl, p_cha.begin() + (i + 1) * mnl, c_cha.begin() + i * (mnl + 1) + 1);
    }

    auto index_data = input.openDataSet("index");

    std::vector<float> mu_t(NW, 0.0f);
    std::vector<std::uint32_t> nw(NW, 0);
    std::vector<std::uint32_t> nl(NW, 0);
    std::vector<float> mus(NW, 0.0f);
    std::vector<float> sig2s(NW, 0.0f);
    std::vector<float> sig2w(NW, 0.0f);

    {
        H5::CompType t(sizeof(float));
        t.insertMember("mu0", 0, float_type);
        index_data.read(mu_t.data(), t);
    }
    {
        H5::CompType t(sizeof(std::uint32_t));
        t.insertMember("l_wave", 0, H5::PredType::NATIVE_UINT32);
        index_data.read(nw.data(), t);
    }
    {
        H5::CompType t(sizeof(std::uint32_t));
        t.insertMember("l_t", 0, H5::PredType::NATIVE_UINT32);
        index_data.read(nl.data(), t);
    }
    {
        H5::CompType t(sizeof(float));
        t.insertMember("mus", 0, float_type);
        index_data.read(mus.data(), t);
    }
    {
        H5::CompType t(sizeof(float));
        t.insertMember("sig2s", 0, float_type);
        index_data.read(sig2s.data(), t);
    }
    {
        H5::CompType t(sizeof(float));
        t.insertMember("sig2w", 0, float_type);
        index_data.read(sig2w.data(), t);
    }

    thrust::device_vector<std::uint32_t> dnw = nw;
    thrust::device_vector<std::uint32_t> dnl = nl;
    thrust::device_vector<float> dcx = A;
    thrust::device_vector<float> dtlist = tlist;
    thrust::device_vector<float> dz = z;
    thrust::device_vector<float> dA = A;
    thrust::device_vector<float> dp_cha = p_cha;
    thrust::device_vector<float> dc_cha = c_cha;
    thrust::device_vector<float> dmu_t = mu_t;
    thrust::device_vector<float> dmus = mus;
    thrust::device_vector<float> dsig2s = sig2s;
    thrust::device_vector<float> dsig2w = sig2w;

    // returns
    thrust::device_vector<std::uint32_t> s0_history(NW * TRIALS, 0.0f);
    thrust::device_vector<float> delta_nu_history(NW * TRIALS, 0.0f);
    thrust::device_vector<float> t0_history(NW * TRIALS, 0.0f);

    // buffers
    thrust::device_vector<float> istar(NW * TRIALS, 0.0f);
    thrust::device_vector<float> home_s(NW * TRIALS, 0.0f);
    thrust::device_vector<float> A_vec(NW * mnw, 0.0f);
    thrust::device_vector<float> c_vec(NW * mnw, 0.0f);
    thrust::device_vector<float> delta_cx(NW * mnw * mnl, 0.0f);
    thrust::device_vector<float> delta_z(NW * mnw, 0.0f);
    thrust::device_vector<float> temp_cx(NW * mnw * mnl, 0.0f);
    thrust::device_vector<float> temp_z(NW * mnw, 0.0f);
    thrust::device_vector<float> wanders(NW * TRIALS, 0.0f);
    thrust::device_vector<float> wts(NW * TRIALS, 0.0f);
    thrust::device_vector<float> accepts(NW * TRIALS, 0.0f);
    thrust::device_vector<float> accts(NW * TRIALS, 0.0f);
    thrust::device_vector<fsmp_step> flip(NW * TRIALS, none);

    constexpr std::size_t BLOCKS = 100;
    std::size_t nslice = div_ceil(NW, BLOCKS);
    for (std::size_t i = 1; i < nslice; i++)
    {
        std::size_t offset = i * BLOCKS;
        std::size_t count = std::min(BLOCKS, NW - offset);
        std::cout << "Starting " << offset << " to " << offset + count - 1 << std::endl;

        flow CUDA_KERNEL(count, 256)(
            count, mnw, mnl,
            dnw.data().get() + offset, dnl.data().get() + offset,
            dcx.data().get() + offset * mnw * mnl, dtlist.data().get() + offset * mnl, dz.data().get() + offset * mnw,
            dmus.data().get() + offset, dsig2s.data().get() + offset, dsig2w.data().get() + offset,
            dA.data().get() + offset * mnw * mnl, dp_cha.data().get() + offset * mnl, dc_cha.data().get() + offset * (mnl + 1),
            dmu_t.data().get() + offset,
            s0_history.data().get() + offset * TRIALS, delta_nu_history.data().get() + offset * TRIALS, t0_history.data().get() + offset * TRIALS,
            istar.data().get() + offset * TRIALS, home_s.data().get() + offset * TRIALS,
            A_vec.data().get() + offset * mnw, c_vec.data().get() + offset * mnw,
            delta_cx.data().get() + offset * mnw * mnl, delta_z.data().get() + offset * mnw,
            temp_cx.data().get() + offset * mnw * mnl, temp_z.data().get() + offset * mnw,
            wanders.data().get() + offset * TRIALS, wts.data().get() + offset * TRIALS,
            accepts.data().get() + offset * TRIALS, accts.data().get() + offset * TRIALS,
            flip.data().get() + offset * TRIALS);

        cudaError_t err = cudaGetLastError();
        assert(err == cudaSuccess);

        err = cudaDeviceSynchronize();
        assert(err == cudaSuccess);
    }

    thrust::host_vector<std::uint32_t> host_s0 = s0_history;
    thrust::host_vector<float> host_nu = delta_nu_history;
    thrust::host_vector<float> host_t0 = t0_history;

    auto output = H5::H5File(output_name, H5F_ACC_TRUNC);

    hsize_t len_history = TRIALS;
    H5::DataSpace space(1, &len_history);

    output.createDataSet("s0", H5::PredType::NATIVE_UINT32, space).write(host_s0.data(), H5::PredType::NATIVE_UINT32, space);
    output.createDataSet("delta_nu", float_type, space).write(host_nu.data(), float_type, space);
    output.createDataSet("t0", float_type, space).write(host_t0.data(), float_type, space);

    return 0;
}
