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

#include "SimDataLinear.hh"

namespace IMFPMaxPhoton {
	extern float Emin;
	extern float Emax;
	extern int NumData;
	extern float InvDelta;
    extern std::vector<float>  DataX;
    extern std::vector<float>  DataY;

    // Linear interpolation. ilow is the `j` index such that x_j <= x < x_{j+1}
    inline float GetValueAt(float xval, int ilow) {
		float denom = IMFPMaxPhoton::DataX[ilow + 1] - IMFPMaxPhoton::DataX[ilow];	
		return (IMFPMaxPhoton::DataY[ilow + 1] - IMFPMaxPhoton::DataY[ilow]) * (xval - IMFPMaxPhoton::DataX[ilow]) / denom 
                + IMFPMaxPhoton::DataY[ilow];
    }
    inline float GetValue(float xval) {
        int ilow = (int)((xval - IMFPMaxPhoton::Emin) * IMFPMaxPhoton::InvDelta);
        ilow = std::max(0, std::min(IMFPMaxPhoton::NumData - 2, ilow));
        return GetValueAt(xval, ilow);
    }
    inline float GetIMFP(float ekin) {
		// make sure that E_min <= ekin < E_max
		const float e = std::min(IMFPMaxPhoton::Emax - 1.0E-6f, std::max(IMFPMaxPhoton::Emin, ekin));
        return std::max(1.0E-20f, GetValue(e));
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
