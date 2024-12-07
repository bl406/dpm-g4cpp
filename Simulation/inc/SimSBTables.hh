#ifndef SimSBTables_HH
#define SimSBTables_HH

//
// M. Novak: 2021
//
// Sampling tables for generating energy transfer values when electron emitts
// bremsstrahlung photons according to the Seltzer-Berger DCS without rejection.
//
// It is assumed, that the required data has been generated and written into
// the `brem_SBDtrData.dat` file previously by using the `SBTableBuilder`.
// This file contains a discrete primary energy grid that is common for all
// materials. Then for each material, a set of sampling tables (`SimLinAliasData`)
// generated at the above discrete primary ecetron energies. All the data
// can be loaded by using the provided `LoadData(dataDir)` method then the
// `SampleEnergyTransfer()` can be used to generate values of energy transferred
// to the emitted photon according to the Seltzer-Berger DCS at the given primary
// electron energy and material without rejection.


#include <vector>
#include <string>

class SimLinAliasData;

namespace SBTables {
	extern __constant__ int NumMaterial;
	extern __constant__ int SamplingTableSize;
	extern __constant__ int NumPrimaryEnergies;
	extern __constant__ float MinPrimaryEnergy;
	extern __constant__ float LogMinPrimaryEnergy;
	extern __constant__ float InvLogDeltaPrimaryEnergy;

	extern cudaArray_t arrXdata;
	extern cudaArray_t arrYdata;
	extern cudaArray_t arrAliasW;
	extern cudaArray_t arrAliasIndx;

	extern cudaTextureObject_t texXdata;
	extern cudaTextureObject_t texYdata;
	extern cudaTextureObject_t texAliasW;
	extern cudaTextureObject_t texAliasIndx;

	extern __device__ cudaTextureObject_t d_texXdata;
	extern __device__ cudaTextureObject_t d_texYdata;
	extern __device__ cudaTextureObject_t d_texAliasW;
	extern __device__ cudaTextureObject_t d_texAliasIndx;

	extern __device__ float Sample(int imat, int penergyindx, float rndm1, float rndm2);
	extern __device__ float SampleEnergyTransfer(float eprim, int imat, float rndm1, float rndm2, float rndm3);
};

class SimSBTables {
public:

  // CTR and DTR
  SimSBTables();
 ~SimSBTables() { CleanTables(); }

  // loads the previously prepared data from the appropriate file locating in
  // the directory specified by the `dataDir` input argument
  void   LoadData(const std::string& dataDir, int verbose);

  // samples values of energy transferred to the emitted gamma according to the
  // Seltzer-Berger DCS at the given `eprim` primary electron energy and material
  // specified by its `imat` index. The three additional input arguments are
  // uniformly random values on [0,1].
  //
  // NOTE: it is assumed that: gamma-cut < eprim < E_max
  double SampleEnergyTransfer(double eprim, int imat, double rndm1, double rndm2, double rndm3);

private:

  void   CleanTables();
  void InitializeTables();

private:

  // Number of materials (each has their own set of sampling table)
  int                   fNumMaterial;
  // The size of the individual sampling tables
  int                   fSamplingTableSize;
  // The number of discrete primary energies at which sampling tables are built.
  int                   fNumPrimaryEnergies;
  // Minimum primary energy from which sampling tables are built
  double                fMinPrimaryEnergy;
  // The logarithm of the minimum primary energy.
  double                fLogMinPrimaryEnergy;
  // Inverse delta log kinetic energy (i.e. bin size in log-scale).
  double                fInvLogDeltaPrimaryEnergy;

  // Define a set of sampling tables for a given material as a new type
  typedef std::vector<SimLinAliasData*> TablesPerMaterial;

  // The collection of sampling tables for all materials
  std::vector<TablesPerMaterial>  fTheTables;

};

#endif // SimSBTables_HH
