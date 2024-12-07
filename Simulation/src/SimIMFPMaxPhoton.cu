#include "SimIMFPMaxPhoton.hh"

#include <cstdio>
#include <iostream>
#include "Utils.h"  

namespace IMFPMaxPhoton {
    __constant__ float Emin;
    __constant__ float Emax;
    __constant__ float InvDelta;

    cudaArray_t array;
    cudaTextureObject_t tex;
    __device__ cudaTextureObject_t d_tex;

    __device__ float GetValue(float xval) {
        float ilow = (xval - IMFPMaxPhoton::Emin) * IMFPMaxPhoton::InvDelta;
        return tex1D<float>(d_tex, ilow + 0.5f);
    }
}

void SimIMFPMaxPhoton::initializeTable(){
    float aux;
	aux = (float)fEmin;
    cudaMemcpyToSymbol(IMFPMaxPhoton::Emin, &aux, sizeof(int));
	aux = (float)fEmax;
	cudaMemcpyToSymbol(IMFPMaxPhoton::Emax, &aux, sizeof(int));
	aux = (float)fData.GetInvDelta();
	cudaMemcpyToSymbol(IMFPMaxPhoton::InvDelta, &aux, sizeof(int));

	std::vector<float> DataY;
    DataY.resize(fData.GetNumData());
	for (int i = 0; i < fData.GetNumData(); ++i) {
		DataY[i] = (float)fData.GetData(i).fY;
	}

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;   
	int NumData = fData.GetNumData();
    initCudaTexture(DataY.data(), &NumData, 1, &texDesc, IMFPMaxPhoton::tex, IMFPMaxPhoton::array);

	cudaMemcpyToSymbol(IMFPMaxPhoton::d_tex, &IMFPMaxPhoton::tex, sizeof(cudaTextureObject_t));

    CudaCheckError();
}

void  SimIMFPMaxPhoton::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/imfp_globalMax.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimIMFPMaxPhoton::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 3 lines are comments
  for (int i=0; i<3; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the electron energy grid
  int numData;
  fscanf(f, "%d\n", &numData);
  if (verbose >0) {
    std::cout << " == Loading global max of total IMFP data: "
              << numData << " discrete values for Linear interpolation. "
              << std::endl;
  }
  // one additional line of comment
  fgets(name, sizeof(name), f);
  // load the fNumData E, tota-IMFP data and fill in the linear interplator
  fData.SetSize(numData);
  for (int i=0; i<numData; ++i) {
    double ekin, val;
    fscanf(f, "%lg %lg", &ekin, &val);
    fData.FillData(i, ekin, val);
    if (i==0)         { fEmin = ekin; }
    if (i==numData-1) { fEmax = ekin; }
  }
  fclose(f);

  initializeTable();
}
