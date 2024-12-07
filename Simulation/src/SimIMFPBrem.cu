#include "SimIMFPBrem.hh"

#include <cstdio>
#include <iostream>
#include "Utils.h"

namespace IMFPBrem {
    __constant__ float Estep;
    __constant__ float Emin;
    __constant__ float Emax;
    __constant__ int ne;
    __constant__ int nmat;

    cudaArray_t array;
    cudaTextureObject_t tex;
    __device__ cudaTextureObject_t d_tex;

    __device__ float GetIMFPPerDensity(float ekin, int imat) {
        // check vacuum case i.e. imat = -1
        if (imat < 0) return 1.0E-20f;
        // make sure that E_min <= ekin < E_max
        const float e = fmin(Emax - 1.0E-6f, fmax(Emin, ekin));
        float index = (e - Emin) / Estep;
        return tex2D<float>(d_tex, index + 0.5f, imat + 0.5f);
    }
};

void SimIMFPBrem::initializeIMFPBremTable()
{
    int ne = 500;
    float Estep = (float)(fEmax - fEmin) / ne;
    float auxilary;
    auxilary = (float)fEmax;
    cudaMemcpyToSymbol(IMFPBrem::Emax, &auxilary, sizeof(float));
	auxilary = (float)fEmin;
    cudaMemcpyToSymbol(IMFPBrem::Emin, &auxilary, sizeof(float));
    cudaMemcpyToSymbol(IMFPBrem::ne, &ne, sizeof(int));
    cudaMemcpyToSymbol(IMFPBrem::nmat, &fNumMaterial, sizeof(int));
	auxilary = (float)Estep;
    cudaMemcpyToSymbol(IMFPBrem::Estep, &auxilary, sizeof(float));

    std::vector<float> table;
    table.resize(ne * fNumMaterial);
    for (int i = 0; i < fNumMaterial; i++) {
        for (int j = 0; j < ne; j++) {
            table[i * ne + j] = (float)GetIMFPPerDensity(double(fEmin + j * Estep), i);
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    initCudaTexture(table.data(), &ne, 1, &texDesc, IMFPBrem::tex, IMFPBrem::array);

	cudaMemcpyToSymbol(IMFPBrem::d_tex, &IMFPBrem::tex, sizeof(cudaTextureObject_t));
}

SimIMFPBrem::SimIMFPBrem () {
  // all will be set at LoadData()
  fNumMaterial = -1;
  fEmin        = -1.;
  fEmax        = -1.;
}

SimIMFPBrem::~SimIMFPBrem() {
  fDataPerMaterial.clear();
}

void  SimIMFPBrem::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/imfp_brem.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimIMFPBrem::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 3 lines are comments
  for (int i=0; i<3; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the electron energy grid and #materials
  int numData;
  fscanf(f, "%d  %d\n", &numData, &fNumMaterial);
  if (verbose > 0) {
    std::cout << " == Loading Brem. IMFP (scalled) data per-material: "
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
        std::cout << "    --- The Brem IMFP data were computed for: " << name;
      }
    }
    // load the fNumData E, IMFP/density data and fill in the Spline interplator
    fDataPerMaterial[imat].SetSize(numData);
    for (int i=0; i<numData; ++i) {
      double ekin, val;
      fscanf(f, "%lg %lg\n", &ekin, &val);
      fDataPerMaterial[imat].FillData(i, ekin, val);
      if (imat==0 && i==0)         { fEmin = ekin; }
      if (imat==0 && i==numData-1) { fEmax = ekin; }
    }
  }
  fclose(f);

  initializeIMFPBremTable();
}
