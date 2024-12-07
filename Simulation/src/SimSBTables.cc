#include "SimSBTables.hh"

#include "SimLinAliasData.hh"

#include <iostream>
#include <cstdio>
#include <cmath>

int SimSBTables::NumMaterial;
int SimSBTables::SamplingTableSize;
int SimSBTables::NumPrimaryEnergies;
float SimSBTables::MinPrimaryEnergy;
float SimSBTables::LogMinPrimaryEnergy;
float SimSBTables::InvLogDeltaPrimaryEnergy;
std::vector<float> SimSBTables::XdataTable;
std::vector<float> SimSBTables::YdataTable;
std::vector<float> SimSBTables::AliasWTable;
std::vector<int> SimSBTables::AliasIndxTable;

void SimSBTables::InitializeTables()
{
	NumMaterial = fNumMaterial;
    SamplingTableSize = fSamplingTableSize;
    NumPrimaryEnergies = fNumPrimaryEnergies;
    MinPrimaryEnergy = (float)fMinPrimaryEnergy;
    LogMinPrimaryEnergy = (float)fLogMinPrimaryEnergy;
    InvLogDeltaPrimaryEnergy = (float)fInvLogDeltaPrimaryEnergy;

    XdataTable.resize(NumMaterial * NumPrimaryEnergies * SamplingTableSize);
    YdataTable.resize(NumMaterial * NumPrimaryEnergies * SamplingTableSize);
    AliasWTable.resize(NumMaterial * NumPrimaryEnergies * SamplingTableSize);
    AliasIndxTable.resize(NumMaterial * NumPrimaryEnergies * SamplingTableSize);

    int index;
    for (int im = 0; im < NumMaterial; ++im) {
        for (int ie = 0; ie < NumPrimaryEnergies; ++ie) {
            for (int is = 0; is < SamplingTableSize; ++is) {
                index = im * NumPrimaryEnergies * SamplingTableSize + ie * SamplingTableSize + is;
                XdataTable[index] = (float)fTheTables[im][ie]->GetOnePoint(is).fXdata;
                YdataTable[index] = (float)fTheTables[im][ie]->GetOnePoint(is).fYdata;
                AliasWTable[index] = (float)fTheTables[im][ie]->GetOnePoint(is).fAliasW;
                AliasIndxTable[index] = fTheTables[im][ie]->GetOnePoint(is).fAliasIndx;
            }
        }
    }
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

float SimSBTables::Sample(int imat, int penergyindx, float rndm1, float rndm2) {
    // get the lower index of the bin by using the alias part
    double rest = rndm1 * (SamplingTableSize - 1);
    int    indxl = (int)(rest);

    if (AliasWTable[imat * NumPrimaryEnergies * SamplingTableSize + penergyindx * SamplingTableSize + indxl] < rest - indxl)
        indxl = AliasIndxTable[imat * NumPrimaryEnergies * SamplingTableSize + penergyindx * SamplingTableSize + indxl];
    // sample value within the selected bin by using linear aprox. of the p.d.f.
    float xval = XdataTable[imat * NumPrimaryEnergies * SamplingTableSize + penergyindx * SamplingTableSize + indxl];
    float xdelta = XdataTable[imat * NumPrimaryEnergies * SamplingTableSize + penergyindx * SamplingTableSize + indxl + 1] - xval;
	float yval = YdataTable[imat * NumPrimaryEnergies * SamplingTableSize + penergyindx * SamplingTableSize + indxl];
    if (yval > 0.0) {
        float dum = (YdataTable[imat * NumPrimaryEnergies * SamplingTableSize + penergyindx * SamplingTableSize + indxl + 1] - yval) / yval;
        if (std::abs(dum) > 0.1)
            return xval - xdelta / dum * (1.0f - std::sqrt(1.0f + rndm2 * dum * (dum + 2.0f)));
        else // use second order Taylor around dum = 0.0
            return xval + rndm2 * xdelta * (1.0f - 0.5f * dum * (rndm2 - 1.0f) * (1.0f + dum * rndm2));
    }
    return xval + xdelta * std::sqrt(rndm2);
}

// it is assumed that: gamma-cut < eprim < E_max
float SimSBTables::SampleEnergyTransfer(float eprim, int imat, float rndm1, float rndm2, float rndm3)
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
