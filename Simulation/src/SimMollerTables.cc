#include "SimMollerTables.hh"

#include "SimLinAliasData.hh"

#include <iostream>
#include <cstdio>
#include <cmath>

int SimMollerTables::SampleTableSize;
int SimMollerTables::NumPrimaryEnergies;
float SimMollerTables::MinPrimaryEnergy;
float SimMollerTables::LogMinPrimaryEnergy;
float SimMollerTables::InvLogDeltaPrimaryEnergy;
std::vector<float> SimMollerTables::XdataTable;
std::vector<float> SimMollerTables::YdataTable;
std::vector<float> SimMollerTables::AliasWTable;
std::vector<int> SimMollerTables::AliasIndxTable;

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
    SampleTableSize = fSamplingTableSize;
    NumPrimaryEnergies = fNumPrimaryEnergies;
    MinPrimaryEnergy = (float)fMinPrimaryEnergy;
    LogMinPrimaryEnergy = (float)fLogMinPrimaryEnergy;
    InvLogDeltaPrimaryEnergy = (float)fInvLogDeltaPrimaryEnergy;

    XdataTable.resize(NumPrimaryEnergies*SampleTableSize);
    YdataTable.resize(NumPrimaryEnergies * SampleTableSize);
    AliasWTable.resize(NumPrimaryEnergies * SampleTableSize);
    AliasIndxTable.resize(NumPrimaryEnergies * SampleTableSize);

    int index;
    for (int ie = 0; ie < NumPrimaryEnergies; ++ie) {
        for (int is = 0; is < SampleTableSize; ++is) {
            index = ie * SampleTableSize + is;
            XdataTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fXdata;
            YdataTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fYdata;
            AliasWTable[index] = (float)fTheTables[ie]->GetOnePoint(is).fAliasW;
            AliasIndxTable[index] = fTheTables[ie]->GetOnePoint(is).fAliasIndx;
        }
    }
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

// produce sample from the represented distribution u
float SimMollerTables::Sample(int penergyindx, float rndm1, float rndm2) {
    // get the lower index of the bin by using the alias part
    float rest = rndm1 * (SampleTableSize - 1);
    int    indxl = (int)(rest);    
    if (AliasWTable[penergyindx * SampleTableSize + indxl] < rest - indxl)
        indxl = AliasIndxTable[penergyindx * SampleTableSize + indxl];
    // sample value within the selected bin by using linear aprox. of the p.d.f.
    float xval = XdataTable[penergyindx * SampleTableSize + indxl];
    float xdelta = XdataTable[penergyindx * SampleTableSize + indxl + 1] - xval;
	float yval = YdataTable[penergyindx * SampleTableSize + indxl];
    if (yval > 0.0f) {
        float dum = (YdataTable[penergyindx * SampleTableSize + indxl + 1] - yval) / yval;
        if (std::abs(dum) > 0.1f)
            return xval - xdelta / dum * (1.0f - std::sqrt(1.0f + rndm2 * dum * (dum + 2.0f)));
        else // use second order Taylor around dum = 0.0
            return xval + rndm2 * xdelta * (1.0f - 0.5f * dum * (rndm2 - 1.0f) * (1.0f + dum * rndm2));
    }
    return xval + xdelta * std::sqrt(rndm2);
}

float SimMollerTables::SampleEnergyTransfer(float eprim, float rndm1, float rndm2, float rndm3) {
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
