#include "Random.hh"

std::mt19937 Random::generator(123);
std::uniform_real_distribution<double> Random::dis(0.0,1.0);

namespace CuRand {
    __device__ curandState* d_states;
    float* RandomArray;
    __device__ float* d_RandomArray;

#define N 65536

    /* 初始化生成器 */
    __global__ void initGenerator(unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &d_states[idx]);
    }

    void initCurand(int blocks, int threads) {
        curandState* states;
        cudaMalloc(&states, blocks * threads * sizeof(curandState));		
		cudaMemcpyToSymbol(d_states, &states, sizeof(curandState*));
        initGenerator << <blocks, threads >> > (time(NULL));

        cudaMalloc(&RandomArray, N*sizeof(float));
        std::vector<float> randoms(N);
        for (size_t i = 0; i < randoms.size(); ++i) {
            randoms[i] = Random::UniformRand();
        }
        cudaMemcpy(RandomArray, randoms.data(), randoms.size() * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(d_RandomArray, &RandomArray, sizeof(float*));
    }

    /* 获得一个在[0, 1)区间内均匀分布的double型变量 */
    __device__ float rand() {
        /*int idx = blockIdx.x * blockDim.x + threadIdx.x;
        return curand_uniform(&d_states[idx]);*/
        static int id = 0;
        if (id >= N) {
            printf("Run out of random values!");			
        }
		return d_RandomArray[id++];
    }
}