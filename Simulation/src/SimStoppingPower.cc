#include "SimStoppingPower.hh"

#include <cstdio>
#include <iostream>
#include "Utils.h"

void SimStoppingPower::initializeStoppingPowerTable()
{
    int ne = 500;
	float Estep = (fEmax - fEmin) / ne;
	cudaMemcpyToSymbol(StoppingPower::Emax, &fEmax, sizeof(float));
	cudaMemcpyToSymbol(StoppingPower::Emin, &fEmin, sizeof(float));
	cudaMemcpyToSymbol(StoppingPower::ne, &ne, sizeof(int));
	cudaMemcpyToSymbol(StoppingPower::nmat, &fNumMaterial, sizeof(int));
	cudaMemcpyToSymbol(StoppingPower::Estep, &Estep, sizeof(float));

    std::vector<float> StoppingPowerTable;
    StoppingPowerTable.resize(ne * fNumMaterial);
    for (int i = 0; i < fNumMaterial; i++) {
        for (int j = 0; j < ne; j++) {
            StoppingPowerTable[i * ne + j] = GetDEDXPerDensity(double(fEmin + j * Estep), i);
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
	initCudaTexture(StoppingPowerTable.data(), &ne, 1, &texDesc, StoppingPower::tex, StoppingPower::array);
}

SimStoppingPower::SimStoppingPower () {
  // all will be set at LoadData()
  fNumMaterial = -1;
  fEmin        = -1.;
  fEmax        = -1.;
}

SimStoppingPower::~SimStoppingPower() {
  fDataPerMaterial.clear();
}

void  SimStoppingPower::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/eloss_rdedx.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimStoppingPower::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 2 lines are comments
  for (int i=0; i<2; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the electron energy grid and #materials
  int numData;
  fscanf(f, "%d  %d\n", &numData, &fNumMaterial);
  if (verbose > 0) {
    std::cout << " == Loading (scalled) restricted dE/dx data per-material: "
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
        std::cout << "    --- The dE/dx data were computed for: " << name;
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

  initializeStoppingPowerTable();
}
