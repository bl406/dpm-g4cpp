#include "SimGSTables.hh"

#include <iostream>
#include <cstdio>
#include <cmath>

int SimGSTables::SamplingTableSize;
float SimGSTables::MinPrimaryEnergy;
float SimGSTables::LogMinPrimaryEnergy;
float SimGSTables::InvLogDeltaPrimaryEnergy;
float SimGSTables::DeltaCum;

std::vector<float> SimGSTables::VarUTable;
std::vector<float> SimGSTables::ParaATable;
std::vector<float> SimGSTables::ParaBTable;
std::vector<float> SimGSTables::TransformParamTable;
std::vector<float> SimGSTables::PrimaryEnergyGridTable;

void SimGSTables::InitializeTables()
{
	SamplingTableSize = fSamplingTableSize;
	MinPrimaryEnergy = (float)fMinPrimaryEnergy;
	LogMinPrimaryEnergy = (float)fLogMinPrimaryEnergy;
	InvLogDeltaPrimaryEnergy = (float)fInvLogDeltaPrimaryEnergy;
	DeltaCum = (float)fDeltaCum;

    TransformParamTable.resize(fNumPrimaryEnergies);
    PrimaryEnergyGridTable.resize(fNumPrimaryEnergies);
    VarUTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
    ParaATable.resize(fNumPrimaryEnergies * fSamplingTableSize);
    ParaBTable.resize(fNumPrimaryEnergies * fSamplingTableSize);
    for (int i = 0; i < fNumPrimaryEnergies; ++i) {
        TransformParamTable[i] = (float)fTheTables[i]->fTransformParam;
        PrimaryEnergyGridTable[i] = (float)fPrimaryEnergyGrid[i];

        for (int j = 0; j < fSamplingTableSize; ++j){
            VarUTable[i * fSamplingTableSize + j] = (float)fTheTables[i]->fGSTable[j].fVarU;
            ParaATable[i * fSamplingTableSize + j] = (float)fTheTables[i]->fGSTable[j].fParmA;
            ParaBTable[i * fSamplingTableSize + j] = (float)fTheTables[i]->fGSTable[j].fParmB;
        }
    }
}

SimGSTables::SimGSTables() {
  // all members will be set when loading the data from the file
  fSamplingTableSize        = -1;
  fNumPrimaryEnergies       = -1;
  fMinPrimaryEnergy         = -1.;
  fLogMinPrimaryEnergy      = -1.;
  fInvLogDeltaPrimaryEnergy = -1.;
  fDeltaCum                 = -1.;
}

void SimGSTables::LoadData(const std::string& dataDir, int verbose) {
  char name[512];
  sprintf(name, "%s/el_GSDtrData.dat", dataDir.c_str());
  FILE* f = fopen(name, "r");
  if (!f) {
    std::cerr << " *** ERROR SimGSTables::LoadData: \n"
              << "     file = " << name << " not found! "
              << std::endl;
    exit(EXIT_FAILURE);
  }
  // first 4 lines are comments
  for (int i=0; i<4; ++i) { fgets(name, sizeof(name), f); }
  // load the size of the primary energy grid and the individual tables first
  double ddum;
  fscanf(f, "%d %lg %lg %d\n", &fNumPrimaryEnergies, &ddum, &ddum, &fSamplingTableSize);
  if (verbose > 0) {
    std::cout << " == Loading GS tables: "
              << fNumPrimaryEnergies << " tables with a size of "
              << fSamplingTableSize << " each."
              << std::endl;
  }
  fDeltaCum = 1.0/(fSamplingTableSize-1);
  // skipp the next 6 lines that are comments
  for (int i=0; i<6; ++i) {
    fgets(name, sizeof(name), f);
    if (i==4 && verbose>0) {
      std::cout << "    --- The GS angular data were computed for: " << name;
    }
  }
  // clean the tables if any
  CleanTables();
  fTheTables.resize(fNumPrimaryEnergies, nullptr);
  fPrimaryEnergyGrid.resize(fNumPrimaryEnergies);
  // load each primary energies and at each primary energy the corresponding table
  for (int ie=0; ie<fNumPrimaryEnergies; ++ie) {
    // load the table index, primary particle kinetic energy and transformation paramater values
    int    idum;
    double transPar;
    fscanf(f, "%d %lg %lg", &idum, &ddum, &transPar);
    fPrimaryEnergyGrid[ie] = ddum;
    if (ie==0) {
      // this is the electron-cut that is also the e-/e+ absorption energy
      fMinPrimaryEnergy    = ddum;
      fLogMinPrimaryEnergy = std::log(ddum);
    }
    if (ie==1) {
      fInvLogDeltaPrimaryEnergy = 1./(std::log(ddum)-fLogMinPrimaryEnergy);
    }
    // construct a sampling table, load the data and fill in the sampling table
    fTheTables[ie] = new OneGSTable();
    fTheTables[ie]->fTransformParam = transPar;
    fTheTables[ie]->fGSTable.resize(fSamplingTableSize);
    for (int is=0; is<fSamplingTableSize; ++is) {
      double u, a, b;
      fscanf(f, "%lg %lg %lg", &u, &a, &b);
      fTheTables[ie]->fGSTable[is].fVarU  = u;
      fTheTables[ie]->fGSTable[is].fParmA = a;
      fTheTables[ie]->fGSTable[is].fParmB = b;
    }
  }
  fclose(f);

  InitializeTables();
}

float SimGSTables::SampleAngularDeflection(float eprim, float rndm1, float rndm2) {
    // determine electron energy lower grid point and sample if that or one above is used now
    float lpenergy = std::log(eprim);
    float phigher = (lpenergy - LogMinPrimaryEnergy) * InvLogDeltaPrimaryEnergy;
    int penergyindx = (int)phigher;
    // keep the lower index of the energy bin
    const int ielow = penergyindx;
    phigher -= penergyindx;
    if (rndm1 < phigher) {
        ++penergyindx;
    }
    // should always be fine if electron-cut < eprim < E_max but make sure
  //  penergyindx      = std::min(fNumPrimaryEnergies-1, penergyindx);
    // sample the transformed variable \xi
    
    // lower index of the (common) discrete cumulative bin and the residual fraction
    const int    indxl = (int)(rndm2 / DeltaCum);
    const float resid = rndm2 - indxl * DeltaCum;
    // compute value `u` by using ratin based numerical inversion
    const float  parA = ParaATable[penergyindx* SamplingTableSize+indxl];
    const float  parB = ParaBTable[penergyindx * SamplingTableSize + indxl];
    const float    u0 = VarUTable[penergyindx * SamplingTableSize + indxl];
    const float    u1 = VarUTable[penergyindx * SamplingTableSize + indxl + 1];
    const float  dum1 = (1.0f + parA + parB) * DeltaCum * resid;
    const float  dum2 = DeltaCum * DeltaCum + parA * DeltaCum * resid + parB * resid * resid;
    const float  theU = u0 + dum1 / dum2 * (u1 - u0);
    // transform back the sampled `u` to `mu(u)` using the transformation parameter `a`
    // mu(u) = 1 - 2au/[1-u+a] as given by Eq.(34)
    // interpolate (linearly) the transformation parameter to E
    const float a0 = TransformParamTable[ielow];
    const float a1 = TransformParamTable[ielow+1];
    const float e0 = PrimaryEnergyGridTable[ielow];
    const float e1 = PrimaryEnergyGridTable[ielow + 1];

    const float parTransf = (a1 - a0) / (e1 - e0) * (eprim - e0) + a0;
    return 1.f - 2.f * parTransf * theU / (1.f - theU + parTransf);
}

// it is assumed that the `eprim` electron energy: electron-cut < eprim <E_max
double SimGSTables::SampleAngularDeflection(double eprim, double rndm1, double rndm2) {
  // determine electron energy lower grid point and sample if that or one above is used now
  double lpenergy  = std::log(eprim);
  double phigher   = (lpenergy-fLogMinPrimaryEnergy)*fInvLogDeltaPrimaryEnergy;
  int penergyindx  = (int) phigher;
  // keep the lower index of the energy bin
  const int ielow  = penergyindx;
  phigher         -= penergyindx;
  if (rndm1<phigher) {
    ++penergyindx;
  }
  // should always be fine if electron-cut < eprim < E_max but make sure
//  penergyindx      = std::min(fNumPrimaryEnergies-1, penergyindx);
  // sample the transformed variable \xi
  const OneGSTable* theGSTable = fTheTables[penergyindx];
  // lower index of the (common) discrete cumulative bin and the residual fraction
  const int    indxl = (int)(rndm2/fDeltaCum);
  const double resid = rndm2-indxl*fDeltaCum;
  // compute value `u` by using ratin based numerical inversion
  const double  parA = theGSTable->fGSTable[indxl].fParmA;
  const double  parB = theGSTable->fGSTable[indxl].fParmB;
  const double    u0 = theGSTable->fGSTable[indxl].fVarU;
  const double    u1 = theGSTable->fGSTable[indxl+1].fVarU;
  const double  dum1 = (1.0 + parA + parB) * fDeltaCum * resid;
  const double  dum2 = fDeltaCum * fDeltaCum + parA * fDeltaCum * resid + parB * resid * resid;
  const double  theU = u0 + dum1 / dum2 * (u1 - u0);
  // transform back the sampled `u` to `mu(u)` using the transformation parameter `a`
  // mu(u) = 1 - 2au/[1-u+a] as given by Eq.(34)
  // interpolate (linearly) the transformation parameter to E
  const double a0 = fTheTables[ielow]->fTransformParam;
  const double a1 = fTheTables[ielow+1]->fTransformParam;
  const double e0 = fPrimaryEnergyGrid[ielow];
  const double e1 = fPrimaryEnergyGrid[ielow+1];
  const double parTransf = (a1-a0)/(e1-e0)*(eprim-e0)+a0;
  return 1.-2.*parTransf*theU/(1.-theU+parTransf);
}


void SimGSTables::CleanTables() {
  for (std::size_t i=0; i<fTheTables.size(); ++i) {
    if (fTheTables[i]) delete fTheTables[i];
  }
  fTheTables.clear();
}
