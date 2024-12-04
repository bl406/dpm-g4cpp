#include "SimIMFPPhoton.hh"

#include <cstdio>
#include <iostream>
#include <assert.h>

#include "utils.h"

namespace IMFPTotal {
	int NumMaterial;
    int NumData;
    float Emin;
    float Emax;
    float InvDelta;
    std::vector<float>  DataX;
    std::vector<float>  DataY;
};

namespace IMFPCompton {
    int NumMaterial;
    int NumData;
    float Emin;
    float Emax;
    float InvDelta;
    std::vector<float>  DataX;
    std::vector<float>  DataY;
};

namespace IMFPPairProd {
    int NumMaterial;
    int NumData;
    float Emin;
    float Emax;
    float InvDelta;
    std::vector<float>  DataX;
    std::vector<float>  DataY;
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

    IMFPTotal::NumMaterial = fNumMaterial;
	IMFPTotal::NumData = fDataPerMaterial[0].GetNumData();
	IMFPTotal::Emin = (float)fEmin;
	IMFPTotal::Emax = (float)fEmax;
	IMFPTotal::InvDelta = (float)fDataPerMaterial[0].GetInvDelta();
	IMFPTotal::DataX.resize(IMFPTotal::NumMaterial * IMFPTotal::NumData);
	IMFPTotal::DataY.resize(IMFPTotal::NumMaterial * IMFPTotal::NumData);
	for (int i = 0; i < fNumMaterial; ++i) {
		for (int j = 0; j < IMFPTotal::NumData; ++j) {
			IMFPTotal::DataX[i * IMFPTotal::NumData + j] = (float)fDataPerMaterial[i].GetData(j).fX;
			IMFPTotal::DataY[i * IMFPTotal::NumData + j] = (float)fDataPerMaterial[i].GetData(j).fY;
		}
	}
}

void SimIMFPPhoton::InitializeIMFPComptonTable() {
    DataValidation();

    IMFPCompton::NumMaterial = fNumMaterial;
    IMFPCompton::NumData = fDataPerMaterial[0].GetNumData();
    IMFPCompton::Emin = (float)fEmin;
    IMFPCompton::Emax = (float)fEmax;
    IMFPCompton::InvDelta = (float)fDataPerMaterial[0].GetInvDelta();
    IMFPCompton::DataX.resize(IMFPCompton::NumMaterial * IMFPCompton::NumData);
    IMFPCompton::DataY.resize(IMFPCompton::NumMaterial * IMFPCompton::NumData);
    for (int i = 0; i < fNumMaterial; ++i) {
        for (int j = 0; j < IMFPCompton::NumData; ++j) {
            IMFPCompton::DataX[i * IMFPCompton::NumData + j] = (float)fDataPerMaterial[i].GetData(j).fX;
            IMFPCompton::DataY[i * IMFPCompton::NumData + j] = (float)fDataPerMaterial[i].GetData(j).fY;
        }
    }
}

void SimIMFPPhoton::InitializeIMFPPairProd() {
    DataValidation();

    IMFPPairProd::NumMaterial = fNumMaterial;
    IMFPPairProd::NumData = fDataPerMaterial[0].GetNumData();
    IMFPPairProd::Emin = (float)fEmin;
    IMFPPairProd::Emax = (float)fEmax;
    IMFPPairProd::InvDelta = (float)fDataPerMaterial[0].GetInvDelta();
    IMFPPairProd::DataX.resize(IMFPPairProd::NumMaterial * IMFPPairProd::NumData);
    IMFPPairProd::DataY.resize(IMFPPairProd::NumMaterial * IMFPPairProd::NumData);
    for (int i = 0; i < fNumMaterial; ++i) {
        for (int j = 0; j < IMFPPairProd::NumData; ++j) {
            IMFPPairProd::DataX[i * IMFPPairProd::NumData + j] = (float)fDataPerMaterial[i].GetData(j).fX;
            IMFPPairProd::DataY[i * IMFPPairProd::NumData + j] = (float)fDataPerMaterial[i].GetData(j).fY;
        }
    }
}
