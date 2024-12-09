#include "SimIMFPMoller.hh"

#include <cstdio>
#include <iostream>
#include "Utils.h"

namespace IMFPMoller {
	__constant__ float Estep;
	__constant__ float Emin;
	__constant__ float Emax;
	__constant__ int ne;

	cudaArray_t array;
	cudaTextureObject_t tex;
	__device__ cudaTextureObject_t d_tex;
}

void SimIMFPMoller::initializeIMFPMollerTable() {
	int ne = 500;   
	float Estep = (float)(fEmax - fEmin) / ne;
    float aux;

	cudaMemcpyToSymbol(IMFPMoller::ne, &ne, sizeof(int));
	aux = (float)fEmin;
	cudaMemcpyToSymbol(IMFPMoller::Emin, &aux, sizeof(float));
	aux = (float)fEmax;
	cudaMemcpyToSymbol(IMFPMoller::Emax, &aux, sizeof(float));
	aux = (float)Estep;
	cudaMemcpyToSymbol(IMFPMoller::Estep, &aux, sizeof(float));

	std::vector<float> IMFPMollerTable;
	IMFPMollerTable.resize(ne);
	for (int j = 0; j < ne; j++) {
		IMFPMollerTable[j] = (float)GetIMFPPerDensity(double(fEmin + j * Estep));
	}

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.normalizedCoords = 0;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.addressMode[0] = cudaAddressModeClamp;
	initCudaTexture(IMFPMollerTable.data(), &ne, 1, &texDesc, IMFPMoller::tex, IMFPMoller::array);
	cudaMemcpyToSymbol(IMFPMoller::d_tex, &IMFPMoller::tex, sizeof(cudaTextureObject_t));
}

void  SimIMFPMoller::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/imfp_moller.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimIMFPMoller::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 3 lines are comments
  for (int i=0; i<3; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the electron energy grid
  int numData;
  fscanf(f, "%d\n", &numData);
  if (verbose > 0) {
    std::cout << " == Loading Moller IMFP (scalled) data: "
              << numData << " discrete values for Spline interpolation. "
              << std::endl;
  }
  // first 4 lines are comments
  for (int i=0; i<4; ++i) {
    fgets(name, sizeof(name), f);
    if (i==2 && verbose>0) {
      std::cout << "    --- The Moller IMFP data were computed for: " << name;
    }
  }
  // load the fNumData E, IMFP/density data and fill in the Spline interplator
  fData.SetSize(numData);
  for (int i=0; i<numData; ++i) {
    double ekin, val;
    fscanf(f, "%lg %lg", &ekin, &val);
    fData.FillData(i, ekin, val);
    if (i==0)         { fEmin = ekin; }
    if (i==numData-1) { fEmax = ekin; }
  }
  fclose(f);

  initializeIMFPMollerTable();
}