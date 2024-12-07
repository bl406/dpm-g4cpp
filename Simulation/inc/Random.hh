
#ifndef Random_HH
#define Random_HH

#include <cstdlib>
#include <random>
#include <curand_kernel.h>

class Random{
public:
  static float UniformRand(){
    return float(dis(generator));
  }

public:
   static std::mt19937 generator;
   static std::uniform_real_distribution<double> dis;
};

namespace CuRand {
	extern __device__ curandState* d_states;
	extern void initCurand();
	/* 获得一个在[0, 1)区间内均匀分布的float型变量 */
	__device__ extern float rand();
}

#endif // Random_HH
