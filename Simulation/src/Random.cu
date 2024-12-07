#include "Random.hh"

std::mt19937 Random::generator(123);
std::uniform_real_distribution<double> Random::dis(0.0,1.0);

namespace CuRand {
    __device__ curandState* d_states;

    /* ��ʼ�������� */
    __global__ void initGenerator(unsigned long long seed) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        curand_init(seed, idx, 0, &d_states[idx]);
    }

    void initCurand(int blocks, int threads) {
        curandState* states;
        cudaMalloc(&states, blocks * threads * sizeof(curandState));		
		cudaMemcpyToSymbol(d_states, &states, sizeof(curandState*));
        initGenerator << <blocks, threads >> > (time(NULL));
    }

    /* ���һ����[0, 1)�����ھ��ȷֲ���double�ͱ��� */
    __device__ float rand() {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        return curand_uniform(&d_states[idx]);
    }
}