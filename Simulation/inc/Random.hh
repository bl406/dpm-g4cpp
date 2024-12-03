
#ifndef Random_HH
#define Random_HH

#include <cstdlib>
#include <random>


class Random{
public:
  static float UniformRand(){
    return float(dis(generator));
  }

public:
   static std::mt19937 generator;
   static std::uniform_real_distribution<double> dis;



};

#endif // Random_HH
