#ifndef SimIMFPMaxPhoton_HH
#define SimIMFPMaxPhoton_HH

//
// M. Novak: 2021
//
// Utility class to load maximum total IMFP (macroscopic cross section) and
// provide (linearly) interpolated values at run time for photons. The maximum
// of the total IMFP is taken at each photon kinetic energies over all possible
// materials.
//
// The maximum total IMFP is loaded from file that has been previously prepared
// by using the `InitPhotonData::InitMXsecData()` method. Note, that the maximum
// total IMFP is not scalled by the material density so it's in [1/mm] units.
// It can be used directly to sample the step length of the photon to the next
// interaction point in the global geometry, where a possible delta (i.e. noting)
// interaction is also included beyond the Compton, Pair-production and
// Photoelectric effects. This is the so-called delta-interaction or Woodcock
// trick for accelerating photon transport across inhomogeneous media.

#include <vector>
#include <string>
#include <cuda_runtime_api.h>
#include "SimDataLinear.hh"

namespace IMFPMaxPhoton {
	extern __constant__ float Emin;
	extern __constant__ float Emax;
	extern __constant__ float InvDelta;

	extern cudaArray_t array;
	extern cudaTextureObject_t tex;
    extern __device__ cudaTextureObject_t d_tex;

    inline __device__ float GetValue(float xval) {
        float ilow = (xval - IMFPMaxPhoton::Emin) * IMFPMaxPhoton::InvDelta;
		return tex1D<float>(d_tex, ilow + 0.5f);
    }
    inline __device__ float GetIMFP(float ekin) {
		// make sure that E_min <= ekin < E_max
		const float e = fmin(IMFPMaxPhoton::Emax - 1.0E-6f, fmax(IMFPMaxPhoton::Emin, ekin));
        return fmax(1.0E-20f, GetValue(e));
    }
};

class SimIMFPMaxPhoton {

public:

  SimIMFPMaxPhoton() {}
 ~SimIMFPMaxPhoton() {}

  void  LoadData(const std::string& dataDir, int verbose);

  // the global maximum inverse MFP in [1/mm] units
  double GetIMFP(double ekin) {
    // make sure that E_min <= ekin < E_max
    const double e = std::min(fEmax-1.0E-6, std::max(fEmin, ekin));
    return std::max(1.0E-20, fData.GetValue(e));
  }
 /* double GetIMFP(double ekin, int ilow) {
    return std::max(1.0E-20, fData.GetValueAt(ekin, ilow));
  }*/


private:
	void initializeTable();

  // store the min/max kinetic enrgy values and the correspnding IMFP values
  double          fEmin;
  double          fEmax;

  // the IMFP data ready for the run-time linear interpolation
  SimDataLinear   fData;
};

#endif // SimIMFPMaxPhoton_HH
