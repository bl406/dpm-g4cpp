#include "SimSBTables.hh"

#include "SimLinAliasData.hh"

#include <iostream>
#include <cstdio>
#include <cmath>
#include "Utils.h"

namespace SBTables {
    __constant__ int NumMaterial;
    __constant__ int SamplingTableSize;
    __constant__ int NumPrimaryEnergies;
    __constant__ float MinPrimaryEnergy;
    __constant__ float LogMinPrimaryEnergy;
    __constant__ float InvLogDeltaPrimaryEnergy;

    cudaArray_t arrXdata;
    cudaArray_t arrYdata;
    cudaArray_t arrAliasW;
    cudaArray_t arrAliasIndx;

    cudaTextureObject_t texXdata;
    cudaTextureObject_t texYdata;
    cudaTextureObject_t texAliasW;
    cudaTextureObject_t texAliasIndx;

    __device__ cudaTextureObject_t d_texXdata;
    __device__ cudaTextureObject_t d_texYdata;
    __device__ cudaTextureObject_t d_texAliasW;
    __device__ cudaTextureObject_t d_texAliasIndx;

    __device__ float Sample(int imat, int penergyindx, float rndm1, float rndm2) {
        // get the lower index of the bin by using the alias part
        float rest = rndm1 * (SamplingTableSize - 1);
        int    indxl = (int)(rest);

        if (tex3D<float>(d_texAliasW, indxl + 0.5f, penergyindx + 0.5f, imat + 0.5f) < rest - indxl)
            indxl = tex3D<int>(d_texAliasIndx, indxl + 0.5f, penergyindx + 0.5f, imat + 0.5f);

        // sample value within the selected bin by using linear aprox. of the p.d.f.
        float xval = tex3D<float>(d_texXdata, indxl + 0.5f, penergyindx + 0.5f, imat + 0.5f);
        float xdelta = tex3D<float>(d_texXdata, indxl + 1 + 0.5f, penergyindx + 0.5f, imat + 0.5f) - xval;
        float yval = tex3D<float>(d_texYdata, indxl + 0.5f, penergyindx + 0.5f, imat + 0.5f);
        if (yval > 0.0f) {
            float dum = (tex3D<float>(d_texYdata, indxl + 1 + 0.5f, penergyindx + 0.5f, imat + 0.5f) - yval) / yval;
            if (std::abs(dum) > 0.1f)
                return xval - xdelta / dum * (1.0f - std::sqrt(1.0f + rndm2 * dum * (dum + 2.0f)));
            else // use second order Taylor around dum = 0.0
                return xval + rndm2 * xdelta * (1.0f - 0.5f * dum * (rndm2 - 1.0f) * (1.0f + dum * rndm2));
        }
        return xval + xdelta * std::sqrt(rndm2);
    }

    // it is assumed that: gamma-cut < eprim < E_max
    __device__ float SampleEnergyTransfer(float eprim, int imat, float rndm1, float rndm2, float rndm3)
    {
        if (imat < 0) return 0.0;
        // determine the primary electron energy lower grid point and sample if that or one above is used now
        float lpenergy = std::log(eprim);
        float phigher = (lpenergy - LogMinPrimaryEnergy) * InvLogDeltaPrimaryEnergy;
        int penergyindx = (int)phigher;
        phigher -= penergyindx;
        if (rndm1 < phigher) {
            ++penergyindx;
        }
        // should always be fine if gamma-cut < eprim < E_max but make sure
      //  penergyindx       = std::min(fNumPrimaryEnergies-2, penergyindx);
        // sample the transformed variable
        const float     xi = Sample(imat, penergyindx, rndm2, rndm3);
        // transform it back to kappa then to gamma energy (fMinPrimaryEnergy = gcut
        // and fLogMinPrimaryEnergy = log(gcut) so log(kappac) = log(gcut/eprim) =
        // = fLogMinPrimaryEnergy - lpenergy = -(lpenergy - fLogMinPrimaryEnergy) that
        // is computed above but keep it visible here)
        const float kappac = MinPrimaryEnergy / eprim;
        const float kappa = kappac * std::exp(-xi * (LogMinPrimaryEnergy - lpenergy));
        return kappa * eprim;
    }

}


void SimSBTables::InitializeTables()
{
    float auxilary;
	cudaMemcpyToSymbol(SBTables::NumMaterial, &fNumMaterial, sizeof(int));
	cudaMemcpyToSymbol(SBTables::SamplingTableSize, &fSamplingTableSize, sizeof(int));
	cudaMemcpyToSymbol(SBTables::NumPrimaryEnergies, &fNumPrimaryEnergies, sizeof(int));
    auxilary = (float)fMinPrimaryEnergy;
	cudaMemcpyToSymbol(SBTables::MinPrimaryEnergy, &auxilary, sizeof(float));
	auxilary = (float)fLogMinPrimaryEnergy;
	cudaMemcpyToSymbol(SBTables::LogMinPrimaryEnergy, &auxilary, sizeof(float));
	auxilary = (float)fInvLogDeltaPrimaryEnergy;
	cudaMemcpyToSymbol(SBTables::InvLogDeltaPrimaryEnergy, &auxilary, sizeof(float));


    std::vector<float> XdataTable;
    std::vector<float> YdataTable;
    std::vector<float> AliasWTable;
    std::vector<int> AliasIndxTable;
    XdataTable.resize(fNumMaterial * fNumPrimaryEnergies * fSamplingTableSize);
    YdataTable.resize(fNumMaterial * fNumPrimaryEnergies * fSamplingTableSize);
    AliasWTable.resize(fNumMaterial * fNumPrimaryEnergies * fSamplingTableSize);
    AliasIndxTable.resize(fNumMaterial * fNumPrimaryEnergies * fSamplingTableSize);

    int index;
    for (int im = 0; im < fNumMaterial; ++im) {
        for (int ie = 0; ie < fNumPrimaryEnergies; ++ie) {
            for (int is = 0; is < fSamplingTableSize; ++is) {
                index = im * fNumPrimaryEnergies * fSamplingTableSize + ie * fSamplingTableSize + is;
                XdataTable[index] = (float)fTheTables[im][ie]->GetOnePoint(is).fXdata;
                YdataTable[index] = (float)fTheTables[im][ie]->GetOnePoint(is).fYdata;
                AliasWTable[index] = (float)fTheTables[im][ie]->GetOnePoint(is).fAliasW;
                AliasIndxTable[index] = fTheTables[im][ie]->GetOnePoint(is).fAliasIndx;
            }
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;

	int size[3] = { fSamplingTableSize, fNumPrimaryEnergies, fNumMaterial };
	initCudaTexture(XdataTable.data(), size, 3, &texDesc, SBTables::texXdata, SBTables::arrXdata);
	initCudaTexture(YdataTable.data(), size, 3, &texDesc, SBTables::texYdata, SBTables::arrYdata);
	initCudaTexture(AliasWTable.data(), size, 3, &texDesc, SBTables::texAliasW, SBTables::arrAliasW);
	initCudaTexture(AliasIndxTable.data(), size, 3, &texDesc, SBTables::texAliasIndx, SBTables::arrAliasIndx);
    cudaMemcpyToSymbol(SBTables::d_texXdata, &SBTables::texXdata, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(SBTables::d_texYdata, &SBTables::texYdata, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(SBTables::d_texAliasW, &SBTables::texAliasW, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(SBTables::d_texAliasIndx, &SBTables::texAliasIndx, sizeof(cudaTextureObject_t));
}


SimSBTables::SimSBTables() {
  // all members will be set when loading the data from the file
  fNumMaterial              = -1;
  fSamplingTableSize        = -1;
  fNumPrimaryEnergies       = -1;
  fMinPrimaryEnergy         = -1.;
  fLogMinPrimaryEnergy      = -1.;
  fInvLogDeltaPrimaryEnergy = -1.;
}


void SimSBTables::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/brem_SBDtrData.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimSBTables::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  //
  // First 5 lines are comments
  for (int i=0; i<5; ++i) { fgets(name, sizeof(name), f); }
  //
  // Load the size of the primary energy grid, the size of the individual tables
  // and number of materials first
  fscanf(f, "%d %d %d", &fNumPrimaryEnergies, &fSamplingTableSize, &fNumMaterial);
  if (verbose > 0) {
    std::cout << " == Loading Seltzer-Berger tables: for each of the "
              << fNumMaterial << " individual material, there are "
              << fNumPrimaryEnergies << " tables with a size of "
              << fSamplingTableSize << " each."
              << std::endl;
  }
  //
  // Loading the discrete primary electron energy grid now: only some related
  // information is used during the sampling so we take only those
  for (int ie=0; ie<fNumPrimaryEnergies; ++ie) {
    double ddum;
    fscanf(f, "%lg\n", &ddum);
    if (ie==0) {
      // this is the gamma-cut that is also the gamma absorption energy
      fMinPrimaryEnergy    = ddum;
      fLogMinPrimaryEnergy = std::log(ddum);
    }
    if (ie==1) {
      fInvLogDeltaPrimaryEnergy = 1./(std::log(ddum)-fLogMinPrimaryEnergy);
    }
  }
  //
  // Loading the tables for each of the materials:
  //
  // Clean the tables if any, resize the top container according to the number
  // of materials then load the set of tables for each materials
  CleanTables();
  fTheTables.resize(fNumMaterial);
  for (int im=0; im<fNumMaterial; ++im) {
    // Skipp 3 lines that contains only the current material name as comment
    for (int i=0; i<3; ++i) {
      fgets(name, sizeof(name), f);
      if (i==1 && verbose>0) {
        std::cout << "    --- loading SB sampling tables for " << name;
      }
    }
    // resize the TableForAMaterial for this material in order to store the
    // fNumPrimaryEnergies sampling tables (4*fSamplingTableSize data each)
    fTheTables[im].resize(fNumPrimaryEnergies, nullptr);
    for (int ie=0; ie<fNumPrimaryEnergies; ++ie) {
      SimLinAliasData* aTable = new SimLinAliasData(fSamplingTableSize);
      for (int is=0; is<fSamplingTableSize; ++is) {
        double xdata, ydata, aliasw;
        int    aliasi;
        fscanf(f, "%lg %lg %lg %d\n", &xdata, &ydata, &aliasw, &aliasi);
        aTable->FillData(is, xdata, ydata, aliasw, aliasi);
      }
      fTheTables[im][ie] = aTable;
    }
  }
  fclose(f);

  InitializeTables();
}

// it is assumed that: gamma-cut < eprim < E_max
double SimSBTables::SampleEnergyTransfer(double eprim, int imat, double rndm1, double rndm2, double rndm3) {
  if (imat<0) return 0.0;
  // determine the primary electron energy lower grid point and sample if that or one above is used now
  double lpenergy   = std::log(eprim);
  double phigher    = (lpenergy-fLogMinPrimaryEnergy)*fInvLogDeltaPrimaryEnergy;
  int penergyindx   = (int) phigher;
  phigher          -= penergyindx;
  if (rndm1<phigher) {
    ++penergyindx;
  }
  // should always be fine if gamma-cut < eprim < E_max but make sure
//  penergyindx       = std::min(fNumPrimaryEnergies-2, penergyindx);
  // sample the transformed variable
  const double     xi = fTheTables[imat][penergyindx]->Sample(rndm2, rndm3);
  // transform it back to kappa then to gamma energy (fMinPrimaryEnergy = gcut
  // and fLogMinPrimaryEnergy = log(gcut) so log(kappac) = log(gcut/eprim) =
  // = fLogMinPrimaryEnergy - lpenergy = -(lpenergy - fLogMinPrimaryEnergy) that
  // is computed above but keep it visible here)
  const double kappac = fMinPrimaryEnergy/eprim;
  const double kappa  = kappac*std::exp(-xi*(fLogMinPrimaryEnergy - lpenergy));
  return kappa*eprim;
}


void SimSBTables::CleanTables() {
  for (std::size_t im=0; im<fTheTables.size(); ++im) {
    for (std::size_t it=0; it<fTheTables[im].size(); ++it) {
      if (fTheTables[im][it]) delete fTheTables[im][it];
    }
    fTheTables[im].clear();
  }
  fTheTables.clear();
}
