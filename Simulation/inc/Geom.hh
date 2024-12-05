
#ifndef Geom_HH
#define Geom_HH

//
// M. Novak: 2021
//
// A simply voxelised geometry representation for the DPM like simulation
// development with some pre-defined geometrical configurations.
//
// The geometry is described by a set of boxes with configurable size given at
// construction time. The target starts with a box centered around the r(0,0,0)
// location with voxel index of (0,0,0). Then developed by shifting this box
// along the +z as well as along the perpendicular xy plane by the box-size.
// This makes possible the fast computation of the voxel/box (ix,iy,iz) index
// triplet based on the (rx,ry,rz) location: the voxel indices are the integer
// shift numbers along the x,y and z axes. The minimum iz voxel index is 0 since
// the target starts at `-half-box-size` and supposed to be hit by the primary
// particle from the left miving to the (0,0,1) direction i.e. toward to +z.
//
// Some pre-defined geometrical configurations are available and can be selected
// by the `geomIndex` at construction time:
//
//  - geomIndex =  0  : homogeneous material with index 0 (also the default)
//
//  - geomIndex =  1  : 0 - 1  cm --> material index 0
//                      1 - 3  cm --> material index 1
//                      3 -       --> material index 0
//
//  - geomIndex =  2  : 0 - 2  cm --> material index 0
//                      2 - 4  cm --> material index 1
//                      4 -       --> material index 0
//
//  - geomIndex =  3  : 0 - 1   cm --> material index 0
//                      1 - 1.5 cm --> material index 1
//                    1.5 - 2.5 cm --> material index 2
//                    2.5 -     cm --> material index 0
//
//
//  NOTE: maximum extent of the geometry is set to 10 cm in each direction except
//        the -z direction along which (half of) the voxel-(box)-size determies
//        the extent of the geometry. Tracking is terminated if the particle goes
//        further than the extent. This is achived by returning `DistanceToBoundary`
//        of -1. Furhermore, the material index of voxel out of the geometry is
//        supposed to be vacuum and calling `GetMaterialIndex` with such a
//        voxel index triplet results in a material index of -1.
//        Note, all these are just for accelerating the simulation and terminate
//        the simulation history when the particle left the geometry and entered
//        into the vacuum
//
//

#include "SimMaterialData.hh"

#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace Geometry {
    extern cudaTextureObject_t texDensity;
    extern cudaArray_t arrDensity;
    extern __device__ cudaTextureObject_t d_texDensity;
    extern __constant__ float LBox;
    extern __constant__ float InvLBox;
    extern __constant__ float LHalfBox;
    extern __constant__ float Tolerance = 1.0E-4f;;
    extern __constant__ float Extent = 100.f;
    extern __constant__ int PreDefinedGeomIndex;

    extern __constant__ float ElectronCut;
    extern __constant__ float GammaCut;

    extern float* EdepHist;
    extern __device__ float *d_EdepHist;
	extern float* StepHist;
    extern __device__ float* d_StepHist;

	// Computes the distance to the volume boundary from the given `rLocation`
	// location along the given `vDirection` direction. When the input `rLocation`
	// is within `0.5*kTolerance` from a volume boundary, then it first updated by
	// moving the point to the next voxel then the distance to the boundary is
	// computed in this new volume. The `iVoxel` voxel index triplet is always set
	// as well.
	__device__  float DistanceToBoundary(float* rLocation, float* dDirection, int* iVoxel);

    // Returns with the material index based on the input `iVoxel` voxel index
    // triplet.
    //
    // NOTE: it is assumed now (only for the sake of simplicity) that each voxel
    //       is built up from a single material, i.e. no material mixing. However,
    //       the simulation do not make any use of this assumption. Therefore,
    //       by changing `GetMaterialIndex` such that it returns the index of the
    //       main material of the voxel while the `GetVoxelMaterialDensity` returns
    //       whatever fractional density values (in [g/cm3]) won't require any change
    //       in the simulation part.
    __device__  int    GetMaterialIndex(int* iVoxel);

    // Returns with the material density of the voxel specified by its `iVoxel`
    // voxel index triplet input ragument.
    inline __device__ float GetVoxelMaterialDensity(int* iVoxel);/* {
        const int imat = GetMaterialIndex(iVoxel);
        return imat > -1 ? tex1D<float>(d_texDensity, imat) : 1.0E-40f;
    }*/
    inline __device__ float GetVoxelMaterialDensity(int imat);/* {
        return imat > -1 ? tex1D<float>(d_texDensity, imat) : 1.0E-40f;
    }*/

	__device__ void Score(float edep, int iz);
}

class Geom {
    
public:

  // Constructor with the `lbox` voxel box size and maximum extent of the
  // geometry as input raguments. (the latter has a defult 10.0 [cm] value).
  Geom (float lbox, SimMaterialData* matData, int geomIndex=0);

  // Computes the distance to the volume boundary from the given `rLocation`
  // location along the given `vDirection` direction. When the input `rLocation`
  // is within `0.5*kTolerance` from a volume boundary, then it first updated by
  // moving the point to the next voxel then the distance to the boundary is
  // computed in this new volume. The `iVoxel` voxel index triplet is always set
  // as well.
  float DistanceToBoundary(float* rLocation, float* dDirection, int* iVoxel);


  // Returns with the material index based on the input `iVoxel` voxel index
  // triplet.
  //
  // NOTE: it is assumed now (only for the sake of simplicity) that each voxel
  //       is built up from a single material, i.e. no material mixing. However,
  //       the simulation do not make any use of this assumption. Therefore,
  //       by changing `GetMaterialIndex` such that it returns the index of the
  //       main material of the voxel while the `GetVoxelMaterialDensity` returns
  //       whatever fractional density values (in [g/cm3]) won't require any change
  //       in the simulation part.
  int    GetMaterialIndex(int* iVoxel);

  // Returns with the material density of the voxel specified by its `iVoxel`
  // voxel index triplet input ragument.
  float GetVoxelMaterialDensity(int* iVoxel) {
      const int imat = GetMaterialIndex(iVoxel);
      return imat > -1 ? fMaterialData->fMaterialDensity[imat] : 1.0E-40f;
  }
  float GetVoxelMaterialDensity(int imat) {
      return imat > -1 ? fMaterialData->fMaterialDensity[imat] : 1.0E-40f;
  }

  // adds the `edep` energy deposit to the voxel/box with z index of `iz`
  void   Score(float edep, int iz);

  // writes the histograms into the `fname` file
  void   Write(const std::string& fname, int nprimaries);

private:

    void Initialize();

  // tolerance of the boundary computation [mm]
  const float         kTolerance = 1.0E-4f;
  // maximum extent of the geometry (except in the -z direction) in  [mm]
  const float         kExtent    = 100.0f;

  // index of the predefined geometry
  int                  fPreDefinedGeomIndex;

  // lenght of the box (cube for the simplicity)
  float               fLBox;
  float               fInvLBox;
  float               fLHalfBox;

  // poiner to the material data object set at construction (used only for the
  // material density in [g/cm3] in `GetVoxelMaterialDensity`).
  SimMaterialData*     fMaterialData;

  // energy deposit and step number histograms: along the depth (i.e. along +z)
  std::vector<float>  fEdepHist;
  std::vector<float>  fStepHist;

};

#endif //Geom_HH
