#include "SimIMFPPhoton.hh"

#include <cstdio>
#include <iostream>
#include <assert.h>

#include "utils.h"

namespace IMFPTotal{
    cudaTextureObject_t tex;
    cudaArray_t array;
    __device__ cudaTextureObject_t d_tex;
};

namespace IMFPCompton {
    cudaTextureObject_t tex;
    cudaArray_t array;
    __device__ cudaTextureObject_t d_tex;
};

namespace IMFPPairProd {
    cudaTextureObject_t tex;
    cudaArray_t array;
    __device__ cudaTextureObject_t d_tex;
};


SimIMFPPhoton::SimIMFPPhoton (int type) {
  // all will be set at LoadData()
  fNumMaterial = -1;
  fType        = type;
  fEmin        = -1.;
  fEmax        = -1.;
}

SimIMFPPhoton::~SimIMFPPhoton() {
  fDataPerMaterial.clear();
}

void  SimIMFPPhoton::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  const std::string strType = (fType == 0) ? "total" : ( fType == 1 ? "compton" : "pairp" );
  sprintf(name, "%s/imfp_%s.dat", dataDir.c_str(), strType.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimIMFPPhoton::LoadData: \n"
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
    std::cout << " == Loading '" << strType << "' IMFP (scalled) data per-material: "
              << numData << " discrete values for Linear interpolation at each of the "
              << fNumMaterial << " different materials."
              << std::endl;
  }
  // skip one line
  fgets(name, sizeof(name), f);
  // allocate space for Linear-interpolation data and load data for each materials
  fDataPerMaterial.resize(fNumMaterial);
  for (int imat=0; imat<fNumMaterial; ++imat) {
    for (int i=0; i<3; ++i) {
      fgets(name, sizeof(name), f);
      if (i==1 && verbose>0) {
        std::cout << "    --- The IMFP data were computed for: " << name;
      }
    }
    // load the fNumData E, tota-IMFP/density data and fill in the Linear interplator
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
}

void SimIMFPPhoton::DataValidation() {
    std::vector<float> vec;
    for (int i = 0; i < fNumMaterial; ++i) {
        vec.push_back((float)fDataPerMaterial[i].GetInvDelta());
    }
    auto meanstd = CalMeanAndStd(vec);
    assert(meanstd[1] < 1.e-8);

    vec.clear();
    for (int i = 0; i < fNumMaterial; ++i) {
        vec.push_back((float)fDataPerMaterial[i].GetNumData());
    }
    meanstd = CalMeanAndStd(vec);
    assert(meanstd[1] < 1.e-8);
}

void SimIMFPPhoton::InitializeIMFPTotalTable() {

    DataValidation();

	float InvDelta = (float)fDataPerMaterial[0].GetInvDelta();
    float auxilary;
	cudaMemcpyToSymbol(IMFPTotal::NumMaterial, &fNumMaterial, sizeof(int));
	auxilary = (float)fEmin;
	cudaMemcpyToSymbol(IMFPTotal::Emin, &auxilary, sizeof(float));
	auxilary = (float)fEmax;
	cudaMemcpyToSymbol(IMFPTotal::Emax, &auxilary, sizeof(float));	
	cudaMemcpyToSymbol(IMFPTotal::InvDelta, &InvDelta, sizeof(float));

    int dim[2] = { fDataPerMaterial[0].GetNumData(), fNumMaterial};

    std::vector<float> DataY;
	DataY.resize(dim[0] * dim[1]);
	for (int i = 0; i < dim[1]; ++i) {
		for (int j = 0; j < dim[0]; ++j) {
			DataY[i * dim[0] + j] = (float)fDataPerMaterial[i].GetData(j).fY;
		}
	}

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    initCudaTexture(DataY.data(), dim, 2, &texDesc, IMFPTotal::tex, IMFPTotal::array);
	cudaMemcpyToSymbol(IMFPTotal::d_tex, &IMFPTotal::tex, sizeof(cudaTextureObject_t));
}

void SimIMFPPhoton::InitializeIMFPComptonTable() {

    DataValidation();

    float InvDelta = (float)fDataPerMaterial[0].GetInvDelta();
	float auxilary;
    cudaMemcpyToSymbol(IMFPCompton::NumMaterial, &fNumMaterial, sizeof(int));
    auxilary = (float)fEmin;
    cudaMemcpyToSymbol(IMFPCompton::Emin, &auxilary, sizeof(float));
    auxilary = (float)fEmax;
    cudaMemcpyToSymbol(IMFPCompton::Emax, &auxilary, sizeof(float));
    cudaMemcpyToSymbol(IMFPCompton::InvDelta, &InvDelta, sizeof(float));

    int dim[2] = { fDataPerMaterial[0].GetNumData(), fNumMaterial };

    std::vector<float> DataY;
    DataY.resize(dim[0] * dim[1]);
    for (int i = 0; i < dim[1]; ++i) {
        for (int j = 0; j < dim[0]; ++j) {
            DataY[i * dim[0] + j] = (float)fDataPerMaterial[i].GetData(j).fY;
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    initCudaTexture(DataY.data(), dim, 2, &texDesc, IMFPCompton::tex, IMFPCompton::array);
    cudaMemcpyToSymbol(IMFPCompton::d_tex, &IMFPCompton::tex, sizeof(cudaTextureObject_t));
}

void SimIMFPPhoton::InitializeIMFPPairProdTable() {

    DataValidation();

    float InvDelta = (float)fDataPerMaterial[0].GetInvDelta();
	float auxilary;
    cudaMemcpyToSymbol(IMFPPairProd::NumMaterial, &fNumMaterial, sizeof(int));
    auxilary = (float)fEmin;
    cudaMemcpyToSymbol(IMFPPairProd::Emin, &auxilary, sizeof(float));
    auxilary = (float)fEmax;
    cudaMemcpyToSymbol(IMFPPairProd::Emax, &auxilary, sizeof(float));
    cudaMemcpyToSymbol(IMFPPairProd::InvDelta, &InvDelta, sizeof(float));

    int dim[2] = { fDataPerMaterial[0].GetNumData(), fNumMaterial };

    std::vector<float> DataY;
    DataY.resize(dim[0] * dim[1]);
    for (int i = 0; i < dim[1]; ++i) {
        for (int j = 0; j < dim[0]; ++j) {
            DataY[i * dim[0] + j] = (float)fDataPerMaterial[i].GetData(j).fY;
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

    initCudaTexture(DataY.data(), dim, 2, &texDesc, IMFPPairProd::tex, IMFPPairProd::array);
    cudaMemcpyToSymbol(IMFPPairProd::d_tex, &IMFPPairProd::tex, sizeof(cudaTextureObject_t));
}
