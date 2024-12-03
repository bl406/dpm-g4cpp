
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
#include <array>

class Geom {
public:
	Geom(SimMaterialData* matData) { fMaterialData = matData; }
	~Geom() {}

	// Computes the distance to the volume boundary from the given `rLocation`
   // location along the given `vDirection` direction. When the input `rLocation`
   // is within `0.5*kTolerance` from a volume boundary, then it first updated by
   // moving the point to the next voxel then the distance to the boundary is
   // computed in this new volume. The `iVoxel` voxel index triplet is always set
   // as well.
	float DistanceToBoundary(double* rLocation, double* dDirection, int* iVoxel);

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
	int    GetMaterialIndex(int* i) {
		return med_indices[i[0] + i[1] * dims[0] + i[2] * dims[1] * dims[0]];
	}

	// Returns with the material density of the voxel specified by its `iVoxel`
	// voxel index triplet input ragument.
	double GetVoxelMaterialDensity(int* iVoxel) {
		const int imat = GetMaterialIndex(iVoxel);
		return imat > -1 ? fMaterialData->fMaterialDensity[imat] : 1.0E-40;
	}
	double GetVoxelMaterialDensity(int imat) {
		return imat > -1 ? fMaterialData->fMaterialDensity[imat] : 1.0E-40;
	}

	void  Score(double edep, int* ivoxel);

	bool LoadEgsPhantom(const std::string& fname);

	void InitScore();

	void InitGeom(int geomIndex);

	// writes the histograms into the `fname` file
	void   Write(const std::string& fname, int nprimaries);
protected:
	
	// tolerance of the boundary computation [mm]
	const float         kTolerance = 1.0E-4;
	// maximum extent of the geometry (except in the -z direction) in  [mm]
	const float         kExtent = 1000.0;
	
	std::vector<int> med_indices;
	std::array<int, 3> dims;
	std::array<float, 3> spacing, inv_spacing;
	std::array<float, 2> xbound, ybound, zbound;

	// poiner to the material data object set at construction (used only for the
	// material density in [g/cm3] in `GetVoxelMaterialDensity`).
	SimMaterialData* fMaterialData;

	float ensrc;               // total energy from source
	std::vector<float> endep;              // 3D dep. energy matrix per batch
	/* The following variables are needed for statistical analysis. Their
	 values are accumulated across the simulation */
	std::vector<float> accum_endep;        // 3D deposited energy matrix
	std::vector<float> accum_endep2;       // 3D square deposited energy
};
#endif //Geom_HH
