#include "SimMaxScatStrength.hh"

#include <cstdio>
#include <iostream>
#include "utils.h"

namespace MaxScatStrength
{
    extern cudaTextureObject_t tex;
    extern cudaArray_t array;
    extern __device__ cudaTextureObject_t d_tex;
};

void SimMaxScatStrength::initializeMaxScatStrengthTable()
{
    int ne = 500;
	float eStep = (float)(fEmax - fEmin) / (ne - 1);
    float aux;
	cudaMemcpyToSymbol(MaxScatStrength::ne, &ne, sizeof(int));
	aux = eStep;
    cudaMemcpyToSymbol(MaxScatStrength::Estep, &aux, sizeof(float));
	aux = fEmin;
    cudaMemcpyToSymbol(MaxScatStrength::Emin, &aux, sizeof(float));
	aux = fEmax;
    cudaMemcpyToSymbol(MaxScatStrength::Emax, &aux, sizeof(float));

	std::vector<float> data;
	data.resize(ne);
	for (int j = 0; j < ne; j++) {
		data[j] = (float)GetMaxScatStrength(double(fEmin + j * eStep));
	}

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;

	initCudaTexture(data.data(), &ne, 1, &texDesc, MaxScatStrength::tex, MaxScatStrength::array);
    cudaMemcpyToSymbol(MaxScatStrength::d_tex, &MaxScatStrength::tex, sizeof(float));
}

void  SimMaxScatStrength::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/el_scatStrength.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimMaxScatStrength::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 4 lines are comments
  for (int i=0; i<4; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the electron energy grid
  int numData;
  fscanf(f, "%d\n", &numData);
  if (verbose > 0) {
    std::cout << " == Loading Maximum Scattering Strength in MSC step data: "
              << numData << " discrete values for Spline interpolation. "
              << std::endl;
  }
  // first 4 lines are comments
  for (int i=0; i<4; ++i) {
    fgets(name, sizeof(name), f);
    if (i==2 && verbose>0) {
      std::cout << "    --- The K_1(E) data were computed for: " << name;
    }
  }
  // load the fNumData E, K_1(E) data and fill in the Spline interplator
  fData.SetSize(numData);
  for (int i=0; i<numData; ++i) {
    double ekin, val;
    fscanf(f, "%lg %lg", &ekin, &val);
    fData.FillData(i, ekin, val);
    if (i==0)         { fEmin = ekin; }
    if (i==numData-1) { fEmax = ekin; }
  }
  fclose(f);

  initializeMaxScatStrengthTable();
}
