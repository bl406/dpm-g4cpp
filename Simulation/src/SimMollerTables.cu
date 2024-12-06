#include "SimMollerTables.hh"

#include "SimLinAliasData.hh"

#include <iostream>
#include <cstdio>
#include <cmath>
#include "Utils.h"

namespace MollerTables {
	cudaArray_t arrXdata;
	cudaArray_t arrYdata;
	cudaArray_t arrAliasW;
	cudaArray_t arrAliasIndx;

    cudaTextureObject_t texXdata, texYdata, texAliasW, texAliasIndx;
    __device__ cudaTextureObject_t d_texXdata, d_texYdata, d_texAliasW, d_texAliasIndx;

	// produce sample from the represented distribution u
	__device__ float Sample(int penergyindx, float rndm1, float rndm2) {
        // get the lower index of the bin by using the alias part
        float rest = rndm1 * (SampleTableSize - 1);
        int    indxl = (int)(rest);
		if (tex2D<float>(d_texAliasW,indxl + 0.5f, penergyindx + 0.5f) < rest - indxl) {
			indxl = tex2D<int>(d_texAliasIndx, indxl + 0.5f, penergyindx + 0.5f);
		}        
        // sample value within the selected bin by using linear aprox. of the p.d.f.
        float xval = tex2D<float>(d_texXdata, indxl+0.5f, penergyindx + 0.5f);
        float xdelta = tex2D<float>(d_texXdata, indxl+1 + 0.5f, penergyindx + 0.5f) - xval;
        float yval = tex2D<float>(d_texYdata, indxl + 0.5f, penergyindx + 0.5f);
        if (yval > 0.0f) {
            float dum = (tex2D<float>(d_texYdata, indxl+1 + 0.5f, penergyindx + 0.5f) - yval) / yval;
            if (std::abs(dum) > 0.1f)
                return xval - xdelta / dum * (1.0f - std::sqrt(1.0f + rndm2 * dum * (dum + 2.0f)));
            else // use second order Taylor around dum = 0.0
                return xval + rndm2 * xdelta * (1.0f - 0.5f * dum * (rndm2 - 1.0f) * (1.0f + dum * rndm2));
        }
        return xval + xdelta * std::sqrt(rndm2);
    }

    __device__ float SampleEnergyTransfer(float eprim, float rndm1, float rndm2, float rndm3) {
        // determine the primary electron energy lower grid point and sample if that or one above is used now
        float lpenergy = std::log(eprim);
        float phigher = (lpenergy - LogMinPrimaryEnergy) * InvLogDeltaPrimaryEnergy;
        int penergyindx = (int)phigher;
        phigher -= penergyindx;
        if (rndm1 < phigher) {
            ++penergyindx;
        }
        // should always be fine if 2 x electron-cut < eprim < E_max (also for e+) but make sure
      //  penergyindx       = std::min(fNumPrimaryEnergies-1, penergyindx);
        // sample the transformed variable xi=[kappa-ln(T_cut/T_0)]/[ln(T_cut/T_0)-ln(T_max/T_0)]
        // where kappa = ln(eps) with eps = T/T_0
        // so xi= [ln(T/T_0)-ln(T_cut/T_0)]/[ln(T_cut/T_0)-ln(T_max/T_0)] that is in [0,1]
        const float   xi = Sample(penergyindx, rndm2, rndm3);
        // fLogMinPrimaryEnergy is log(2*ecut) = log(ecut) - log(0.5)
        const float dum1 = lpenergy - LogMinPrimaryEnergy;
        // return with the sampled kinetic energy transfered to the electron
        // (0.5*fMinPrimaryEnergy is the electron production cut)
        return std::exp(xi * dum1) * 0.5f * MinPrimaryEnergy;
    }

}


SimMollerTables::SimMollerTables() {
  // all members will be set when loading the data from the file
  fSamplingTableSize        = -1;
  fNumPrimaryEnergies       = -1;
  fMinPrimaryEnergy         = -1.;
  fLogMinPrimaryEnergy      = -1.;
  fInvLogDeltaPrimaryEnergy = -1.;
}

void SimMollerTables::InitializeTables()
{
    float auxilary;
    cudaMemcpyToSymbol(MollerTables::SampleTableSize, &fSamplingTableSize, sizeof(int));
	cudaMemcpyToSymbol(MollerTables::NumPrimaryEnergies, &fNumPrimaryEnergies, sizeof(int));
    auxilary = (float)fMinPrimaryEnergy;
	cudaMemcpyToSymbol(MollerTables::MinPrimaryEnergy, &auxilary, sizeof(float));
	auxilary = (float)fLogMinPrimaryEnergy;
	cudaMemcpyToSymbol(MollerTables::LogMinPrimaryEnergy, &auxilary, sizeof(float));
	auxilary = (float)fInvLogDeltaPrimaryEnergy;
	cudaMemcpyToSymbol(MollerTables::InvLogDeltaPrimaryEnergy, &auxilary, sizeof(float));

    std::vector<float> XdataTable;
    std::vector<float> YdataTable;
    std::vector<float> AliasWTable;
    std::vector<int> AliasIndxTable;
    XdataTable.resize(fNumPrimaryEnergies* fSamplingTableSize);
    YdataTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
    AliasWTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
    AliasIndxTable.resize(fNumPrimaryEnergies * fSamplingTableSize);

    int index;
    for (int ie = 0; ie < fNumPrimaryEnergies; ++ie) {
        for (int is = 0; is < fSamplingTableSize; ++is) {
            index = ie * fSamplingTableSize + is;
            XdataTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fXdata;
            YdataTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fYdata;
            AliasWTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fAliasW;
            AliasIndxTable[index] = fTheTables[ie]->GetOnePoint(is).fAliasIndx;
        }
    }

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;

	int size[2] = { fSamplingTableSize, fNumPrimaryEnergies };
    initCudaTexture(XdataTable.data(), size, 2, &texDesc, MollerTables::texXdata, MollerTables::arrXdata);
    initCudaTexture(YdataTable.data(), size, 2, &texDesc, MollerTables::texYdata, MollerTables::arrYdata);
    initCudaTexture(AliasWTable.data(), size, 2, &texDesc, MollerTables::texAliasW, MollerTables::arrAliasW);
    initCudaTexture(AliasIndxTable.data(), size, 2, &texDesc, MollerTables::texAliasIndx, MollerTables::arrAliasIndx);
}

void SimMollerTables::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/ioni_MollerDtrData.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimMollerTables::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 5 lines are comments
  for (int i=0; i<5; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the primary energy grid and the individual tables first
  fscanf(f, "%d %d", &fNumPrimaryEnergies, &fSamplingTableSize);
  if (verbose >0) {
    std::cout << " == Loading Moller tables: "
              << fNumPrimaryEnergies << " tables with a size of "
              << fSamplingTableSize << " each."
              << std::endl;
  }
  // clean tables if any
  CleanTables();
  fTheTables.resize(fNumPrimaryEnergies, nullptr);
  // load each primary energies and at each primary energy the corresponding table
  for (int ie=0; ie<fNumPrimaryEnergies; ++ie) {
    // load the primary particle kinetic energy value and use if it's needed
    double ddum;
    fscanf(f, "%lg", &ddum);
    if (ie==0) {
      // this is 2x electron-cut
      fMinPrimaryEnergy    = ddum;
      fLogMinPrimaryEnergy = std::log(ddum);
    }
    if (ie==1) {
      fInvLogDeltaPrimaryEnergy = 1./(std::log(ddum)-fLogMinPrimaryEnergy);
    }
    // construct a sampling table, load the data and fill in the sampling table
    fTheTables[ie] = new SimLinAliasData(fSamplingTableSize);
    for (int is=0; is<fSamplingTableSize; ++is) {
      double xdata, ydata, aliasw;
      int    aliasi;
      fscanf(f, "%lg %lg %lg %d", &xdata, &ydata, &aliasw, &aliasi);
      fTheTables[ie]->FillData(is, xdata, ydata, aliasw, aliasi);
    }
  }
  fclose(f);

  InitializeTables();
}

// it is assumed that: 2 x electron-cut < eprim < E_max (also for e+)
double SimMollerTables::SampleEnergyTransfer(double eprim, double rndm1, double rndm2, double rndm3) {
  // determine the primary electron energy lower grid point and sample if that or one above is used now
  double lpenergy   = std::log(eprim);
  double phigher    = (lpenergy-fLogMinPrimaryEnergy)*fInvLogDeltaPrimaryEnergy;
  int penergyindx   = (int) phigher;
  phigher          -= penergyindx;
  if (rndm1<phigher) {
    ++penergyindx;
  }
  // should always be fine if 2 x electron-cut < eprim < E_max (also for e+) but make sure
//  penergyindx       = std::min(fNumPrimaryEnergies-1, penergyindx);
  // sample the transformed variable xi=[kappa-ln(T_cut/T_0)]/[ln(T_cut/T_0)-ln(T_max/T_0)]
  // where kappa = ln(eps) with eps = T/T_0
  // so xi= [ln(T/T_0)-ln(T_cut/T_0)]/[ln(T_cut/T_0)-ln(T_max/T_0)] that is in [0,1]
  const double   xi = fTheTables[penergyindx]->Sample(rndm2, rndm3);
  // fLogMinPrimaryEnergy is log(2*ecut) = log(ecut) - log(0.5)
  const double dum1 = lpenergy - fLogMinPrimaryEnergy;
  // return with the sampled kinetic energy transfered to the electron
  // (0.5*fMinPrimaryEnergy is the electron production cut)
  return std::exp(xi*dum1)*0.5*fMinPrimaryEnergy;
}


void SimMollerTables::CleanTables() {
  for (std::size_t i=0; i<fTheTables.size(); ++i) {
    if (fTheTables[i]) delete fTheTables[i];
  }
  fTheTables.clear();
}
