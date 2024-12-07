#include "SimKNTables.hh"

#include "SimLinAliasData.hh"

#include <iostream>
#include <cstdio>
#include <cmath>

namespace KNTables {
    int SamplingTableSize;
    int NumPrimaryEnergies;
    float MinPrimaryEnergy;
    float LogMinPrimaryEnergy;
    float InvLogDeltaPrimaryEnergy;
    std::vector<float> Xdata;
    std::vector<float> Ydata;
    std::vector<float> AliasW;
    std::vector<int> AliasIndex;

    // method provided to produce samples according to the represented distribution
    // `rndm1` and `rndm2` are uniformly random values on [0,1]
    inline float Sample(int ienergy, float rndm1, float rndm2) {
        // get the lower index of the bin by using the alias part
        float rest = rndm1 * (SamplingTableSize - 1);
        int    indxl = (int)(rest);
        
        if (AliasW[ienergy * SamplingTableSize + indxl] < rest - indxl)
            indxl = AliasIndex[ienergy * SamplingTableSize + indxl];
        // sample value within the selected bin by using linear aprox. of the p.d.f.
        float xval = Xdata[ienergy * SamplingTableSize + indxl];
        float xdelta = Xdata[ienergy * SamplingTableSize + indxl + 1] - xval;
		float yval = Ydata[ienergy * SamplingTableSize + indxl];
        if (yval > 0.0f) {
            float dum = (Ydata[ienergy * SamplingTableSize + indxl + 1] - yval) / yval;
            if (std::abs(dum) > 0.1)
                return xval - xdelta / dum * (1.0f - std::sqrt(1.0f + rndm2 * dum * (dum + 2.0f)));
            else // use second order Taylor around dum = 0.0
                return xval + rndm2 * xdelta * (1.0f - 0.5f * dum * (rndm2 - 1.0f) * (1.0f + dum * rndm2));
        }
        return xval + xdelta * std::sqrt(rndm2);
    }

    // it is assumed that: gamma-cut < egamma < E_max
    float SampleEnergyTransfer(float egamma, float rndm1, float rndm2, float rndm3) {
        const float kEMC2 = 0.510991f;
        // determine primary photon energy lower grid point and sample if that or one above is used now
        float lpenergy = std::log(egamma);
        float phigher = (lpenergy - LogMinPrimaryEnergy) * InvLogDeltaPrimaryEnergy;
        int penergyindx = (int)phigher;
        phigher -= penergyindx;
        if (rndm1 < phigher) {
            ++penergyindx;
        }
        // should always be fine if gamma-cut < egamma < E_max but make sure
      //  penergyindx      = std::min(fNumPrimaryEnergies-1, penergyindx);
        // sample the transformed variable xi=[\alpha-ln(ep)]/\alpha (where \alpha=ln(1/(1+2\kappa)))
        // that is in [0,1] when ep is in [ep_min=1/(1+2\kappa),ep_max=1] (that limits comes from energy and momentum
        // conservation in case of scattering on free electron at rest).
        // where ep = E_1/E_0 and kappa = E_0/(mc^2)
        float xi = Sample(penergyindx, rndm2, rndm3);
        // transform it back to eps = E_1/E_0
        // \epsion(\xi) = \exp[ \alpha(1-\xi) ] = \exp [\ln(1+2\kappa)(\xi-1)]
        float kappa = egamma / kEMC2;
        return std::exp(std::log(1.f + 2.f * kappa) * (xi - 1.f)); // eps = E_1/E_0
    }
};

SimKNTables::SimKNTables() {
  // all members will be set when loading the data from the file
  fSamplingTableSize        = -1;
  fNumPrimaryEnergies       = -1;
  fMinPrimaryEnergy         = -1.;
  fLogMinPrimaryEnergy      = -1.;
  fInvLogDeltaPrimaryEnergy = -1.;
}

	KNTables::XdataTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
	KNTables::YdataTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
	KNTables::AliasWTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
	KNTables::AliasIndxTable.resize(fNumPrimaryEnergies * fSamplingTableSize);

	int index;
	for (int ie = 0; ie < fNumPrimaryEnergies; ++ie) {
		for (int is = 0; is < fSamplingTableSize; ++is) {
			index = ie * fSamplingTableSize + is;
			KNTables::XdataTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fXdata;
			KNTables::YdataTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fYdata;
			KNTables::AliasWTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fAliasW;
			KNTables::AliasIndxTable[index] = fTheTables[ie]->GetOnePoint(is).fAliasIndx;
		}
	}
}

void SimKNTables::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/compton_KNDtrData.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimKNTables::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 5 lines are comments
  for (int i=0; i<5; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the primary energy grid and the individual tables first
  fscanf(f, "%d %d", &fNumPrimaryEnergies, &fSamplingTableSize);
  if (verbose > 0) {
    std::cout << " == Loading Klein-Nishina tables: "
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
      // this is the gamma-cut that is also the gamma absorption energy
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

// it is assumed that: gamma-cut < egamma < E_max
double SimKNTables::SampleEnergyTransfer(double egamma, double rndm1, double rndm2, double rndm3) {
  const double kEMC2 = 0.510991;
  // determine primary photon energy lower grid point and sample if that or one above is used now
  double lpenergy  = std::log(egamma);
  double phigher   = (lpenergy-fLogMinPrimaryEnergy)*fInvLogDeltaPrimaryEnergy;
  int penergyindx  = (int) phigher;
  phigher         -= penergyindx;
  if (rndm1<phigher) {
    ++penergyindx;
  }
  // should always be fine if gamma-cut < egamma < E_max but make sure
//  penergyindx      = std::min(fNumPrimaryEnergies-1, penergyindx);
  // sample the transformed variable xi=[\alpha-ln(ep)]/\alpha (where \alpha=ln(1/(1+2\kappa)))
  // that is in [0,1] when ep is in [ep_min=1/(1+2\kappa),ep_max=1] (that limits comes from energy and momentum
  // conservation in case of scattering on free electron at rest).
  // where ep = E_1/E_0 and kappa = E_0/(mc^2)
  double xi = fTheTables[penergyindx]->Sample(rndm2, rndm3);
  // transform it back to eps = E_1/E_0
  // \epsion(\xi) = \exp[ \alpha(1-\xi) ] = \exp [\ln(1+2\kappa)(\xi-1)]
  double kappa = egamma/kEMC2;
  return std::exp(std::log(1.+2.*kappa)*(xi-1.)); // eps = E_1/E_0
}


void SimKNTables::CleanTables() {
  for (std::size_t i=0; i<fTheTables.size(); ++i) {
    if (fTheTables[i]) delete fTheTables[i];
  }
  fTheTables.clear();
}

void SimKNTables::InitializeTables() {
	KNTables::NumPrimaryEnergies = fNumPrimaryEnergies;
	KNTables::SamplingTableSize = fSamplingTableSize;
	KNTables::MinPrimaryEnergy = (float)fMinPrimaryEnergy;
	KNTables::LogMinPrimaryEnergy = (float)fLogMinPrimaryEnergy;
	KNTables::InvLogDeltaPrimaryEnergy = (float)fInvLogDeltaPrimaryEnergy;
	KNTables::Xdata.resize(KNTables::NumPrimaryEnergies * KNTables::SamplingTableSize);
	KNTables::Ydata.resize(KNTables::NumPrimaryEnergies * KNTables::SamplingTableSize);
	KNTables::AliasW.resize(KNTables::NumPrimaryEnergies * KNTables::SamplingTableSize);
	KNTables::AliasIndex.resize(KNTables::NumPrimaryEnergies * KNTables::SamplingTableSize);
	for (int i = 0; i < fNumPrimaryEnergies; i++) {
		for (int j = 0; j < fSamplingTableSize; j++) {
			auto& point = fTheTables[i]->GetOnePoint(j);
			KNTables::Xdata[i * fSamplingTableSize + j] = (float)point.fXdata;
			KNTables::Ydata[i * fSamplingTableSize + j] = (float)point.fYdata;
			KNTables::AliasW[i * fSamplingTableSize + j] = (float)point.fAliasW;
			KNTables::AliasIndex[i * fSamplingTableSize + j] = point.fAliasIndx;
		}
	}
}