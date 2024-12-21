
#ifndef Geom_HH
#define Geom_HH

#include "SimMaterialData.hh"

#include <array>
#include <vector>
#include <string>

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>


namespace Geometry {
	extern __constant__ float d_Spacing[3];
	extern __constant__ int d_Dim[3];
	extern __constant__ int d_Nvoxels;
	extern int h_Nvoxels;
	extern __constant__ float d_Xbound[2], d_Ybound[2], d_Zbound[2];

    extern cudaTextureObject_t texDensity, texMollerIMFPScaling;
	extern cudaArray_t arrDensity, arrMollerIMFPScaling;
	extern __device__ cudaTextureObject_t d_texDensity, d_texMollerIMFPScaling;
   
    extern __constant__ float d_ElectronCut;
    extern __constant__ float d_GammaCut;

	__device__  float DistanceToBoundary(float* rLocation, float* dDirection, int* iVoxel);

    __device__  int    GetMaterialIndex(int* iVoxel);

    inline __device__ float GetVoxelMaterialDensity(int* iVoxel) {
        const int imat = GetMaterialIndex(iVoxel);
        return imat > -1 ? tex1D<float>(d_texDensity, imat) : 1.0E-40f;
    }
    inline __device__ float GetVoxelMaterialDensity(int imat) {
        return imat > -1 ? tex1D<float>(d_texDensity, imat) : 1.0E-40f;
    }

	inline __device__ float GetVoxelMollerIMFPScaling(int imat) {
		return imat > -1 ? tex1D<float>(d_texMollerIMFPScaling, imat) : 1.0E-40f;
	}

	__device__ void Score(float edep, int* i);
	__global__ void AccumulateEndep();
}

class Geom {

public:
	Geom(SimMaterialData* matData, int geomIndex = 0);
	~Geom() {}

	// writes the histograms into the `fname` file
	void  Write(const std::string& fname, int nprimaries, int nbatch);

	void Initialize();
	void Clear();

	const std::array<float, 2>& GetXbound() { return fXbound; }
	const std::array<float, 2>& GetYbound() { return fYbound; }
	const std::array<float, 2>& GetZbound() { return fZbound; }

	// get fXbounds, fYbounds, fZbounds
	const std::vector<float>& GetXbounds() { return fXbounds; }
	const std::vector<float>& GetYbounds() { return fYbounds; }
	const std::vector<float>& GetZbounds() { return fZbounds; }

	const std::array<int, 3>& GetDims() { return fDims; }
	const std::array<float, 3>& GetSpacing() { return fSpacing; }
private:
	std::array<int, 3> fDims;
	std::array<float, 3> fSpacing;
	std::array<float, 2> fXbound, fYbound, fZbound;
	std::vector<float> fXbounds, fYbounds, fZbounds;

	int fPreDefinedGeomIndex;

	// poiner to the material data object set at construction (used only for the
	// material density in [g/cm3] in `GetVoxelMaterialDensity`).
	SimMaterialData* fMaterialData;
};

#endif //Geom_HH
