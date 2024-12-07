
#include "Geom.hh"

#include <cmath>
#include <cstdio>
#include <iostream>
#include "utils.h"

namespace Geometry {
    cudaTextureObject_t texDensity, texMollerIMFPScaling;
    cudaArray_t arrDensity, arrMollerIMFPScaling;
    __device__ cudaTextureObject_t d_texDensity, d_texMollerIMFPScaling;
    
    __constant__ float LBox;
    __constant__ float InvLBox;
    __constant__ float LHalfBox;
    __constant__ float Tolerance;
    __constant__ float Extent;
    __constant__ int PreDefinedGeomIndex;

    __constant__ float ElectronCut;
    __constant__ float GammaCut;

    float* EdepHist;
    __device__ float* d_EdepHist;
    float* StepHist;
	__device__ float* d_StepHist;

    __device__ float DistanceToBoundary(float* r, float* v, int* i) {
        float kHalfTolerance = 0.5f * Tolerance;
        //
        // Let's say that kExtent is the max extent of our geometry except the -z
        // direction in which it's only half box
        if (std::abs(r[0]) > Extent || std::abs(r[1]) > Extent || r[2] > Extent || r[2] < -LBox) {
            // indicates out of the geometry
            return -1.0f;
        }
        // compute the current x,y and z box/volume index based on the current r(rx,ry,rz)
        // round downward to integer
        i[0] = (int)std::floor((r[0] + LHalfBox) * InvLBox);
        i[1] = (int)std::floor((r[1] + LHalfBox) * InvLBox);
        i[2] = (int)std::floor((r[2] + LHalfBox) * InvLBox);
        //
        // transform the r(rx,ry,rz) into the current local box
        const float trX = r[0] - i[0] * LBox;
        const float trY = r[1] - i[1] * LBox;
        const float trZ = r[2] - i[2] * LBox;
        //
        // compute the distance to boundary in the local box (centered around 0,0,0)
        float pdist = 0.0f;
        float stmp = 0.0f;
        float snext = 1.0E+20f;
        //
        // calculate x
        if (v[0] > 0.0f) {
            pdist = LHalfBox - trX;
            // check if actually this location is on boudnary
            if (pdist < kHalfTolerance) {
                // on boundary: push to the next box/volume, i.e. to the otherside and recompute
                r[0] += Tolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                snext = pdist / v[0];
            }
        }
        else if (v[0] < 0.0f) {
            pdist = LHalfBox + trX;
            if (pdist < kHalfTolerance) {
                // push to the otherside
                r[0] -= Tolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                snext = -pdist / v[0];
            }
        }
        //
        // calcualte y
        if (v[1] > 0.0f) {
            pdist = LHalfBox - trY;
            if (pdist < kHalfTolerance) {
                r[1] += Tolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                stmp = pdist / v[1];
                if (stmp < snext) {
                    snext = stmp;
                }
            }
        }
        else if (v[1] < 0.0f) {
            pdist = LHalfBox + trY;
            if (pdist < kHalfTolerance) {
                r[1] -= Tolerance;
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
        if (v[2] > 0.0f) {
            pdist = LHalfBox - trZ;
            if (pdist < kHalfTolerance) {
                r[2] += Tolerance;
                return DistanceToBoundary(r, v, i);
            }
            else {
                stmp = pdist / v[2];
                if (stmp < snext) {
                    snext = stmp;
                }
            }
        }
        else if (v[2] < 0.0f) {
            pdist = LHalfBox + trZ;
            if (pdist < kHalfTolerance) {
                r[2] -= Tolerance;
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
        const int iz = i[2];
        // vacuum
        if (iz < 0) {
            return -1;
        }
        // iz >= 0
        switch (PreDefinedGeomIndex) {
            // homogeneous material with index 0
        case 0:
            return 0;
            // 0 - 1 cm --> 0; 1 - 3 cm --> 1; 3 - ... --> 0;
        case 1:
            if (iz < 10.0 * InvLBox) { return 0; }
            else
                if (iz < 30.0 * InvLBox) { return 1; }
            return 0;
            // 0 - 2 cm --> 0; 2 - 4 cm --> 1; 4 - ... --> 0;
        case 2:
            if (iz < 20.0 * InvLBox) { return 0; }
            else
                if (iz < 40.0 * InvLBox) { return 1; }
            return 0;
            // 0 - 1 cm --> 0; 1 - 1.5 cm --> 1; 1.5 - 2.5 cm --> 2; 2.5 - ... ->0
        case 3:
            if (iz < 10.0 * InvLBox) { return 0; }
            else
                if (iz < 15.0 * InvLBox) { return 1; }
                else
                    if (iz < 25.0 * InvLBox) { return 2; }
            return 0;
            // default: homogeneous material
        default:
            return 0;
        }
    }

    __device__ void Score(float edep, int iz) {
        if (iz >= 0) {
            const std::size_t indx = (std::size_t)(iz);
            atomicAdd(&d_EdepHist[indx], edep);
            atomicAdd(&d_StepHist[indx], 1.0f);
        }
    }
}


void Geom::Initialize() {
    int dim = fMaterialData->fNumMaterial;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;

    initCudaTexture(fMaterialData->fMaterialDensity.data(), &dim, 1, &texDesc, 
        Geometry::texDensity, Geometry::arrDensity);
    initCudaTexture(fMaterialData->fMollerIMFPScaling.data(), &dim, 1, &texDesc,
        Geometry::texMollerIMFPScaling, Geometry::arrMollerIMFPScaling);

    cudaMemcpyToSymbol(Geometry::d_texDensity, &Geometry::texDensity, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(Geometry::d_texMollerIMFPScaling, &Geometry::texMollerIMFPScaling, sizeof(cudaTextureObject_t));

    float Tolerance = 1.0E-4f;
    float Extent = 100.f;

    cudaMemcpyToSymbol(Geometry::Tolerance, &Tolerance, sizeof(float));
    cudaMemcpyToSymbol(Geometry::Extent, &Extent, sizeof(float));

    cudaMemcpyToSymbol(Geometry::LBox, &fLBox, sizeof(float));
    cudaMemcpyToSymbol(Geometry::InvLBox, &fInvLBox, sizeof(float));
    cudaMemcpyToSymbol(Geometry::LHalfBox, &fLHalfBox, sizeof(float));
    cudaMemcpyToSymbol(Geometry::PreDefinedGeomIndex, &fPreDefinedGeomIndex, sizeof(int));

	cudaMemcpyToSymbol(Geometry::ElectronCut, &fMaterialData->fElectronCut, sizeof(float));
	cudaMemcpyToSymbol(Geometry::GammaCut, &fMaterialData->fGammaCut, sizeof(float));

    cudaMalloc(&Geometry::EdepHist, fEdepHist.size() * sizeof(float));
    cudaMemset(Geometry::EdepHist, 0, fEdepHist.size() * sizeof(float));
    cudaMemcpyToSymbol(Geometry::d_EdepHist, &Geometry::EdepHist, sizeof(float*));

    cudaMalloc(&Geometry::StepHist, fStepHist.size() * sizeof(float));
    cudaMemset(Geometry::StepHist, 0, fStepHist.size() * sizeof(float));
    cudaMemcpyToSymbol(Geometry::d_StepHist, &Geometry::StepHist, sizeof(float*));

	CudaCheckError();
}

Geom::Geom(float lbox, SimMaterialData* matData, int geomIndex)
: fPreDefinedGeomIndex(geomIndex),
  fLBox(lbox),
  fInvLBox(1.f/lbox),
  fLHalfBox(0.5f*lbox),
  fMaterialData(matData) {
  fEdepHist.resize(200,0.0f);
  fStepHist.resize(200,0.0f);
}

float Geom::DistanceToBoundary(float* r, float* v, int* i) {
    static float kHalfTolerance = 0.5f * kTolerance;
    //
    // Let's say that kExtent is the max extent of our geometry except the -z
    // direction in which it's only half box
    if (std::abs(r[0]) > kExtent || std::abs(r[1]) > kExtent || r[2] > kExtent || r[2] < -fLBox) {
        // indicates out of the geometry
        return -1.0;
    }
    // compute the current x,y and z box/volume index based on the current r(rx,ry,rz)
    // round downward to integer
    i[0] = (int)std::floor((r[0] + fLHalfBox) * fInvLBox);
    i[1] = (int)std::floor((r[1] + fLHalfBox) * fInvLBox);
    i[2] = (int)std::floor((r[2] + fLHalfBox) * fInvLBox);
    //
    // transform the r(rx,ry,rz) into the current local box
    const float trX = r[0] - i[0] * fLBox;
    const float trY = r[1] - i[1] * fLBox;
    const float trZ = r[2] - i[2] * fLBox;
    //
    // compute the distance to boundary in the local box (centered around 0,0,0)
    float pdist = 0.0f;
    float stmp = 0.0f;
    float snext = 1.0E+20f;
    //
    // calculate x
    if (v[0] > 0.0) {
        pdist = fLHalfBox - trX;
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
        pdist = fLHalfBox + trX;
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
        pdist = fLHalfBox - trY;
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
        pdist = fLHalfBox + trY;
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
        pdist = fLHalfBox - trZ;
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
        pdist = fLHalfBox + trZ;
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


void Geom::Score(float edep, int iz) {
  if (iz>=0) {
    const std::size_t indx = (std::size_t)(iz);
    if (indx>=fEdepHist.size()) {
      fEdepHist.resize(indx+10);
      fStepHist.resize(indx+10);
    }
    fEdepHist[indx] += edep;
    fStepHist[indx] += 1.0f;
  }
}

int Geom::GetMaterialIndex(int* i) {
    const int iz = i[2];
    // vacuum
    if (iz < 0) {
        return -1;
    }
    // iz >= 0
    switch (fPreDefinedGeomIndex) {
        // homogeneous material with index 0
    case 0:
        return 0;
        // 0 - 1 cm --> 0; 1 - 3 cm --> 1; 3 - ... --> 0;
    case 1:
        if (iz < 10.0 * fInvLBox) { return 0; }
        else
            if (iz < 30.0 * fInvLBox) { return 1; }
        return 0;
        // 0 - 2 cm --> 0; 2 - 4 cm --> 1; 4 - ... --> 0;
    case 2:
        if (iz < 20.0 * fInvLBox) { return 0; }
        else
            if (iz < 40.0 * fInvLBox) { return 1; }
        return 0;
        // 0 - 1 cm --> 0; 1 - 1.5 cm --> 1; 1.5 - 2.5 cm --> 2; 2.5 - ... ->0
    case 3:
        if (iz < 10.0 * fInvLBox) { return 0; }
        else
            if (iz < 15.0 * fInvLBox) { return 1; }
            else
                if (iz < 25.0 * fInvLBox) { return 2; }
        return 0;
        // default: homogeneous material
    default:
        return 0;
    }
}


void Geom::Write(const std::string& fname, int nprimaries) {
  FILE *f = fopen(fname.c_str(),"w");
  if (!f) {
    std::cerr<< " *** ERROR in Geom::Write(): "
             << " file " << fname << " could not be created! "
             << std::endl;
  }
  const float toCm = 0.1f;
  int      sizeHist = (int)fEdepHist.size();
  float       norm = 1.f/(nprimaries*fLBox*toCm);
  float    sumEdep = 0.0f;
  for (int i=0; i<sizeHist; ++i) { sumEdep += fEdepHist[i]; }
  fprintf(f, "# === Mean energy deposit in the target: %13.4e  [MeV/event]\n", sumEdep/nprimaries);
  fprintf(f, "# === Energy deposit as a function of the depth: \n");
  // write the histogram: depth dose [MeV/cm]/density[g/cm3]--> [MeV cm2/g] as a function of depth [cm]
  fprintf(f, "# === Index       Depth [cm]    Edep [MeV cm2/g]  #Simulation-Steps \n");
  int idumy[] = {0,0,0};
  for (int i=0; i<sizeHist; ++i) {
    // set the iz voxel index and get the corresponding material density
    idumy[2] = i;
    float matDensity = GetVoxelMaterialDensity(idumy);
    fprintf(f, " %10d    %13.4e    %13.4e    %13.4e\n", i, (i+0.5f)*fLBox*toCm, fEdepHist[i]*norm/matDensity, fStepHist[i]/nprimaries);
  }
  fclose(f);
  std::cout << " === Energy deposit histogram is written to the file:  " << fname << "\n" << std::endl;
}

