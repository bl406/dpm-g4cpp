
#include "Geom.hh"

#include <cmath>
#include <cstdio>
#include <iostream>
#include "metaio.h"
#include "utils.h"

namespace Geometry {
 
#define kTolerance 1.0E-4f
#define kHalfTolerance (0.5f * kTolerance)

    __constant__ float d_ElectronCut;
    __constant__ float d_GammaCut;

    __constant__ float d_Spacing[3];
    __constant__ int d_Dim[3], d_NVoxels;
    int h_Nvoxels;
    __constant__ float d_Xbound[2], d_Ybound[2], d_Zbound[2];

    float* Endep, * AccumEndep, * AccumEndep2;
    __device__ float* d_Endep, * d_AccumEndep, * d_AccumEndep2;

    cudaTextureObject_t texXbounds, texYbounds, texZbounds;
    cudaArray_t arrXbounds, arrYbounds, arrZbounds;
    __device__ cudaTextureObject_t d_texXbounds, d_texYbounds, d_texZbounds;

    cudaTextureObject_t texDensity, texMatIndex, texMollerIMFPScaling;
    cudaArray_t arrDensity, arrMatIndex, arrMollerIMFPScaling;
    __device__ cudaTextureObject_t d_texDensity, d_texMatIndex, d_texMollerIMFPScaling;

    __constant__ float ElectronCut;
    __constant__ float GammaCut;

    __device__ float DistanceToBoundary(float* r, float* v, int* i) {		
        if (r[0] < d_Xbound[0] || r[0] > d_Xbound[1]
            || r[1] < d_Ybound[0] || r[1] > d_Ybound[1]
            || r[2] < d_Zbound[0] || r[2] > d_Zbound[1]) {
            return -1.0f;
        }

        // compute the current x,y and z box/volume index based on the current r(rx,ry,rz)
        // round downward to integer
        i[0] = (int)std::floor((r[0] - d_Xbound[0]) / d_Spacing[0]);
        i[1] = (int)std::floor((r[1] - d_Ybound[0]) / d_Spacing[1]);
        i[2] = (int)std::floor((r[2] - d_Zbound[0]) / d_Spacing[2]);

        // the particle may lie exatly on the boudary of the phantom which make the index invalid
        if (i[0] < 0 || i[0] > d_Dim[0] || i[1] < 0 || i[1] > d_Dim[1] || i[2] < 0 || i[2] > d_Dim[2]) {
            return -1.0f;
        }

        //
        // transform the r(rx,ry,rz) into the current local box
        const float trX = r[0] - tex1D<float>(d_texXbounds, i[0]);
        const float trY = r[1] - tex1D<float>(d_texYbounds, i[1]);
        const float trZ = r[2] - tex1D<float>(d_texZbounds, i[2]);
        //
        // compute the distance to boundary in the local box
        float pdist = 0.0f;
        float stmp = 0.0f;
        float snext = 1.0E+20f;
        //
        // calculate x
        if (v[0] > 0.0) {
            pdist = d_Spacing[0] - trX;
            // check if actually this location is on boudnary
            if (pdist < kHalfTolerance) {
                // on boundary: push to the next box/volume, i.e. to the otherside and recompute
                r[0] += kTolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                snext = pdist / v[0];
            }
        }
        else if (v[0] < 0.0) {
            pdist = trX;
            if (pdist < kHalfTolerance) {
                // push to the otherside
                r[0] -= kTolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                snext = -pdist / v[0];
            }
        }
        //
        // calcualte y
        if (v[1] > 0.0) {
            pdist = d_Spacing[1] - trY;
            if (pdist < kHalfTolerance) {
                r[1] += kTolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                stmp = pdist / v[1];
                if (stmp < snext) {
                    snext = stmp;
                }
            }
        }
        else if (v[1] < 0.0) {
            pdist = trY;
            if (pdist < kHalfTolerance) {
                r[1] -= kTolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                stmp = -pdist / v[1];
                if (stmp < snext) {
                    snext = stmp;
                }
            }
        }
        //
        // calculate z
        if (v[2] > 0.0) {
            pdist = d_Spacing[2] - trZ;
            if (pdist < kHalfTolerance) {
                r[2] += kTolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                stmp = pdist / v[2];
                if (stmp < snext) {
                    snext = stmp;
                }
            }
        }
        else if (v[2] < 0.0) {
            pdist = trZ;
            if (pdist < kHalfTolerance) {
                r[2] -= kTolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                stmp = -pdist / v[2];
                if (stmp < snext) {
                    snext = stmp;
                }
            }
        }

        return snext;
    }

    __device__ int GetMaterialIndex(int* i) {
		return tex3D<float>(d_texMatIndex, i[0] + 0.5f, i[1] + 0.5f, i[2] + 0.5f);
    }

    __device__ void Score(float edep, int* index) {
        if (index[0] >= 0 && index[0] < d_Dim[0] && 
            index[1] >= 0 && index[1] < d_Dim[1] && 
            index[2] >= 0 && index[2] < d_Dim[2]) {
            int idx = index[0] + index[1] * d_Dim[0] + index[2] * d_Dim[1] * d_Dim[0];
            atomicAdd(d_Endep+idx, edep);
        }
    }

	__global__ void AccumulateEndep() {
		int idx = threadIdx.x + blockIdx.x * blockDim.x;
		if (idx < d_NVoxels) {
			d_AccumEndep[idx] += d_Endep[idx];
			d_AccumEndep2[idx] += d_Endep[idx] * d_Endep[idx];
			d_Endep[idx] = 0.0f;
		}
	}
}


Geom::Geom(SimMaterialData* matData, int geomIndex)
    : fPreDefinedGeomIndex(geomIndex), fMaterialData(matData) {
}


void Geom::Initialize() {
    std::vector<int> fMedIndices;   
    int size;

    //
	// initialize the geometry related data
    //
    switch (fPreDefinedGeomIndex){
    case 0:
    default:
        fSpacing.fill(1.0);

        fXbound = { -100.5f, 100.5f };
        fYbound = { -100.5f, 100.5f };
        fZbound = { -0.5f, 100.5f };

        fDims[0] = static_cast<int>(round((fXbound[1] - fXbound[0]) / fSpacing[0]));
        fDims[1] = static_cast<int>(round((fYbound[1] - fYbound[0]) / fSpacing[1]));
        fDims[2] = static_cast<int>(round((fZbound[1] - fZbound[0]) / fSpacing[2]));

        fMedIndices.resize(fDims[0] * fDims[1] * fDims[2]);
        for (int i = 0; i < fDims[0]; i++) {
            for (int j = 0; j < fDims[1]; j++) {
                for (int k = 0; k < fDims[2]; k++) {
                    int irl = i + j * fDims[0] + k * fDims[1] * fDims[0];                    
                    fMedIndices[irl] = 0;
                }
            }
        }
		
        break;;
    }

    fXbounds.resize(fDims[0] + 1);
    fYbounds.resize(fDims[1] + 1);
    fZbounds.resize(fDims[2] + 1);

    fXbounds[0] = fXbound[0];
    for (int i = 1; i < fDims[0] + 1; i++) {
        fXbounds[i] = fXbounds[i - 1] + fSpacing[0];
    }
    fYbounds[0] = fYbound[0];
    for (int i = 1; i < fDims[1] + 1; i++) {
        fYbounds[i] = fYbounds[i - 1] + fSpacing[1];
    }
    fZbounds[0] = fZbound[0];
    for (int i = 1; i < fDims[2] + 1; i++) {
        fZbounds[i] = fZbounds[i - 1] + fSpacing[2];
    }  

    //
	// copy the geometry related data to the device
    //
    Geometry::h_Nvoxels = fDims[0] * fDims[1] * fDims[2];
	cudaMemcpyToSymbol(Geometry::d_Dim, fDims.data(), 3 * sizeof(int));
	cudaMemcpyToSymbol(Geometry::d_NVoxels, &Geometry::h_Nvoxels, sizeof(int));
	cudaMemcpyToSymbol(Geometry::d_Spacing, fSpacing.data(), 3 * sizeof(float));
	cudaMemcpyToSymbol(Geometry::d_Xbound, fXbound.data(), 2 * sizeof(float));
	cudaMemcpyToSymbol(Geometry::d_Ybound, fYbound.data(), 2 * sizeof(float));
	cudaMemcpyToSymbol(Geometry::d_Zbound, fZbound.data(), 2 * sizeof(float));
	
    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.addressMode[2] = cudaAddressModeClamp;
	size = fDims[0] + 1;  
    initCudaTexture(fXbounds.data(), &size, 1, &texDesc, Geometry::texXbounds, Geometry::arrXbounds);
	size = fDims[1] + 1;
    initCudaTexture(fYbounds.data(), &size, 1, &texDesc, Geometry::texYbounds, Geometry::arrYbounds);
	size = fDims[2] + 1;
    initCudaTexture(fZbounds.data(), &size, 1, &texDesc, Geometry::texZbounds, Geometry::arrZbounds);
	initCudaTexture(fMedIndices.data(), fDims.data(), 3, &texDesc, Geometry::texMatIndex, Geometry::arrMatIndex);

	cudaMemcpyToSymbol(Geometry::d_texXbounds, &Geometry::texXbounds, sizeof(cudaTextureObject_t));
	cudaMemcpyToSymbol(Geometry::d_texYbounds, &Geometry::texYbounds, sizeof(cudaTextureObject_t));
	cudaMemcpyToSymbol(Geometry::d_texZbounds, &Geometry::texZbounds, sizeof(cudaTextureObject_t));
	cudaMemcpyToSymbol(Geometry::d_texMatIndex, &Geometry::texMatIndex, sizeof(cudaTextureObject_t));

	CudaCheckError();

    cudaMalloc(&Geometry::Endep, Geometry::h_Nvoxels * sizeof(float));
    cudaMalloc(&Geometry::AccumEndep, Geometry::h_Nvoxels * sizeof(float));
    cudaMalloc(&Geometry::AccumEndep2, Geometry::h_Nvoxels * sizeof(float));
    cudaMemset(Geometry::Endep, 0, Geometry::h_Nvoxels * sizeof(float));
    cudaMemset(Geometry::AccumEndep, 0, Geometry::h_Nvoxels * sizeof(float));
    cudaMemset(Geometry::AccumEndep2, 0, Geometry::h_Nvoxels * sizeof(float));
	cudaMemcpyToSymbol(Geometry::d_Endep, &Geometry::Endep, sizeof(float*));
	cudaMemcpyToSymbol(Geometry::d_AccumEndep, &Geometry::AccumEndep, sizeof(float*));
	cudaMemcpyToSymbol(Geometry::d_AccumEndep2, &Geometry::AccumEndep2, sizeof(float*));

    //
	// copy the material related data to the device
    //
    cudaMemcpyToSymbol(Geometry::d_ElectronCut, &fMaterialData->fElectronCut, sizeof(float));
    cudaMemcpyToSymbol(Geometry::d_GammaCut, &fMaterialData->fGammaCut, sizeof(float));
	size = (int)fMaterialData->fMaterialDensity.size();
	initCudaTexture(fMaterialData->fMaterialDensity.data(), &size, 1, &texDesc, Geometry::texDensity, Geometry::arrDensity);
    initCudaTexture(fMaterialData->fMollerIMFPScaling.data(), &size, 1, &texDesc, Geometry::texMollerIMFPScaling, Geometry::arrMollerIMFPScaling);
	cudaMemcpyToSymbol(Geometry::d_texDensity, &Geometry::texDensity, sizeof(cudaTextureObject_t));
	cudaMemcpyToSymbol(Geometry::d_texMollerIMFPScaling, &Geometry::texMollerIMFPScaling, sizeof(cudaTextureObject_t));	

    CudaCheckError();
}

void Geom::Clear()
{
	// free the device memory
	cudaFree(Geometry::Endep);
	cudaFree(Geometry::AccumEndep);
	cudaFree(Geometry::AccumEndep2);
	cudaDestroyTextureObject(Geometry::texXbounds);
	cudaDestroyTextureObject(Geometry::texYbounds);
	cudaDestroyTextureObject(Geometry::texZbounds);
	cudaDestroyTextureObject(Geometry::texMatIndex);
	cudaDestroyTextureObject(Geometry::texDensity);
	cudaDestroyTextureObject(Geometry::texMollerIMFPScaling);
	cudaFreeArray(Geometry::arrXbounds);
	cudaFreeArray(Geometry::arrYbounds);
	cudaFreeArray(Geometry::arrZbounds);
	cudaFreeArray(Geometry::arrMatIndex);
	cudaFreeArray(Geometry::arrDensity);
	cudaFreeArray(Geometry::arrMollerIMFPScaling);

}

void Geom::Write(const std::string& fname, int nprimaries, int nbatch) {
    // To do: compute the variance and normalize the accumulated energy deposition
    float endep, endep2, unc_endep;
    int h_Nvoxels = fDims[0] * fDims[1] * fDims[2];

	std::vector<float> fAccumEndep(h_Nvoxels);
	std::vector<float> fAccumEndep2(h_Nvoxels);
	cudaMemcpy(fAccumEndep.data(), Geometry::AccumEndep, h_Nvoxels * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(fAccumEndep2.data(), Geometry::AccumEndep2, h_Nvoxels * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < h_Nvoxels; ++i)
    {
        endep = fAccumEndep[i] / nbatch;
        endep2 = fAccumEndep2[i] / nbatch;
        /* Batch approach uncertainty calculation */
        if (endep != 0.0) {
            unc_endep = endep2 - endep * endep;
            unc_endep /= (nbatch - 1);

            /* Relative uncertainty */
            unc_endep = sqrt(unc_endep) / endep;
        }
        else {
            endep = 0.0;
            unc_endep = 0.9999999f;
        }

        fAccumEndep[i] = endep;
        fAccumEndep2[i] = unc_endep;
    }

    /* Zero dose in air */
    /*for (int i = 0; i < h_Nvoxels; ++i) {
        if (fMedIndices[i] == 0) {
            fAccumEndep[i] = 0.0;
            fAccumEndep2[i] = 0.9999999f;
        }
    }*/

    writeMetaImage(fname + ".dose", fDims, fSpacing, fAccumEndep.data());
    writeMetaImage(fname + ".unc", fDims, fSpacing, fAccumEndep2.data());
}
