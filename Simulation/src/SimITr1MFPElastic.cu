#include "SimITr1MFPElastic.hh"

#include <cstdio>
#include <iostream>
#include "Utils.h"

namespace ITr1MFPElastic{
    cudaArray_t array;
    cudaTextureObject_t tex;
    __device__ cudaTextureObject_t d_tex;
};

void SimITr1MFPElastic::initializeITr1MFPTable(){
    int ne = 500;
    float Estep = (float)(fEmax - fEmin) / ne;
	float auxilary;
    cudaMemcpyToSymbol(ITr1MFPElastic::ne, &ne, sizeof(int));
	auxilary = fEmax;
    cudaMemcpyToSymbol(ITr1MFPElastic::Emax, &auxilary, sizeof(float));
	auxilary = fEmin;
    cudaMemcpyToSymbol(ITr1MFPElastic::Emin, &auxilary, sizeof(float));   
	auxilary = Estep;
    cudaMemcpyToSymbol(ITr1MFPElastic::Estep, &auxilary, sizeof(float));

    std::vector<float> ITr1MFPPerDensityTable;
    ITr1MFPPerDensityTable.resize(ne * fNumMaterial);
    for (int i = 0; i < fNumMaterial; i++) {
        for (int j = 0; j < ne; j++) {
            ITr1MFPPerDensityTable[i * ne + j] = GetITr1MFPPerDensity(double(fEmin + j * Estep), i);
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

	int size[2] = { ne, fNumMaterial };
	initCudaTexture(ITr1MFPPerDensityTable.data(), size, 2, &texDesc, ITr1MFPElastic::tex, ITr1MFPElastic::array);

	cudaMemcpyToSymbol(ITr1MFPElastic::d_tex, &ITr1MFPElastic::tex, sizeof(cudaTextureObject_t));
}

SimITr1MFPElastic::SimITr1MFPElastic() {
  fNumMaterial = -1;
  fEmin        = -1.;
  fEmax        = -1.;
}


void  SimITr1MFPElastic::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/el_itr1mfp.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimITr1MFPElastic::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 4 lines are comments
  for (int i=0; i<4; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the electron energy grid and #materials
  int numData;
  fscanf(f, "%d  %d\n", &numData, &fNumMaterial);
  if (verbose > 0) {
    std::cout << " == Loading Inverse Tr1-MFP (scalled) data: "
              << numData << " discrete values for Spline interpolation at each of the "
              << fNumMaterial << " different materials."
              << std::endl;
  }
  // skip one line
  fgets(name, sizeof(name), f);
  // allocate space for Spline-interpolation data and load data for each materials
  fDataPerMaterial.resize(fNumMaterial);
  for (int imat=0; imat<fNumMaterial; ++imat) {
    for (int i=0; i<3; ++i) {
      fgets(name, sizeof(name), f);
      if (i==1 && verbose>0) {
        std::cout << "    --- The Inverse Tr1-MFP data were computed for: " << name;
      }
    }
    // load the fNumData E, I-Tr1-MFP/density data and fill in the Spline interplator
    fDataPerMaterial[imat].SetSize(numData);
    for (int i=0; i<numData; ++i) {
      double ekin, val, ddum;
      fscanf(f, "%lg %lg %lg\n", &ekin, &ddum, &val);
      fDataPerMaterial[imat].FillData(i, ekin, val);
      if (imat==0 && i==0)         { fEmin = ekin; }
      if (imat==0 && i==numData-1) { fEmax = ekin; }
    }
  }
  fclose(f);

  initializeITr1MFPTable();
}
