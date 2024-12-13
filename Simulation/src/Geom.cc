
#include "Geom.hh"

#include <cmath>
#include <cstdio>
#include <iostream>
#include "metaio.h"

Geom::Geom(float lbox, SimMaterialData* matData, int geomIndex)
    : fPreDefinedGeomIndex(geomIndex),
    fLBox(lbox),
    fInvLBox(1.f / lbox),
    fLHalfBox(0.5f * lbox),
    fMaterialData(matData) {
}

void Geom::InitScore()
{
    fEnsrc = 0.0;
    fEndep.resize(fDims[0] * fDims[1] * fDims[2], 0.f);
    fAccumEndep.resize(fDims[0] * fDims[1] * fDims[2], 0.f);
    fAccumEndep2.resize(fDims[0] * fDims[1] * fDims[2], 0.f);
}

float Geom::DistanceToBoundary(float* r, float* v, int* i) {
    static float kHalfTolerance = 0.5f * kTolerance;

    if (r[0] < fXbound[0] || r[0] > fXbound[1]
        || r[1] < fYbound[0] || r[1] > fYbound[1]
        || r[2] < fZbound[0] || r[2] > fZbound[1]){
        return -1.0f;
    }

    // compute the current x,y and z box/volume index based on the current r(rx,ry,rz)
    // round downward to integer
    i[0] = (int)std::floor((r[0] - fXbound[0]) * fInvSpacing[0]);
    i[1] = (int)std::floor((r[1] - fYbound[0]) * fInvSpacing[1]);
    i[2] = (int)std::floor((r[2] - fZbound[0]) * fInvSpacing[2]);

    // the particle may lie exatly on the boudary of the phantom which make the index invalid
    if (i[0] < 0 || i[0] > fDims[0] || i[1] < 0 || i[1] > fDims[1] || i[2] < 0 || i[2] > fDims[2]) {
		return -1.0f;
    }

    //
    // transform the r(rx,ry,rz) into the current local box
    const float trX = r[0] - fXbounds[i[0]];
    const float trY = r[1] - fYbounds[i[1]];
    const float trZ = r[2] - fZbounds[i[2]];
    //
    // compute the distance to boundary in the local box
    float pdist = 0.0f;
    float stmp = 0.0f;
    float snext = 1.0E+20f;
    //
    // calculate x
    if (v[0] > 0.0) {
        pdist = fSpacing[0] - trX;
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
        pdist = fSpacing[1] - trY;
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
        pdist =  trY;
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
        pdist = fSpacing[2] - trZ;
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

float Geom::DistanceToBoundaryOriginal(float* r, float* v, int* i) {
    static float kHalfTolerance = 0.5f * kTolerance;
    //
    // Let's say that kExtent is the max extent of our geometry except the -z
    // direction in which it's only half box
    if (std::abs(r[0]) > kExtent || std::abs(r[1]) > kExtent || r[2] > kExtent || r[2] < -fLBox) {
        // indicates out of the geometry
        return -1.0f;
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
    if (v[0] > 0.0f) {
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
    else if (v[0] < 0.0f) {
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
    if (v[1] > 0.0f) {
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
    else if (v[1] < 0.0f) {
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
    if (v[2] > 0.0f) {
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
    else if (v[2] < 0.0f) {
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

void Geom::Score(float edep, int* ivoxel) {
    int index = ivoxel[0] + ivoxel[1] * fDims[0] + ivoxel[2] * fDims[1] * fDims[0];
    if(index < fDims[0]* fDims[1]* fDims[2])
        fEndep[index] += edep;
}

int Geom::GetMaterialIndexOriginal(int* i) {
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


void Geom::InitGeom() {
    fSpacing.fill(1.0);
    fInvSpacing.fill(1.0);
    switch (fPreDefinedGeomIndex)
    {
    case 3:
    default:
        fXbound = { -100.5f, 100.5f };
        fYbound = { -100.5f, 100.5f };
        fZbound = { -0.5f, 100.5f };
        break;;
    }

    fDims[0] = static_cast<int>(round((fXbound[1] - fXbound[0]) / fSpacing[0]));
    fDims[1] = static_cast<int>(round((fYbound[1] - fYbound[0]) / fSpacing[1]));
    fDims[2] = static_cast<int>(round((fZbound[1] - fZbound[0]) / fSpacing[2]));

	fXbounds.resize(fDims[0] + 1);
	fYbounds.resize(fDims[1] + 1);
	fZbounds.resize(fDims[2] + 1);

    fXbounds[0] = fXbound[0];
	for (int i = 1; i < fDims[0] + 1; i++) {
		fXbounds[i] = fXbounds[i-1] + fSpacing[0];
	}
	fYbounds[0] = fYbound[0];
	for (int i = 1; i < fDims[1] + 1; i++) {
		fYbounds[i] = fYbounds[i - 1] + fSpacing[1];
	}
	fZbounds[0] = fZbound[0];
	for (int i = 1; i < fDims[2] + 1; i++) {
		fZbounds[i] = fZbounds[i - 1] + fSpacing[2];
	}

    fMedIndices.resize(fDims[0] * fDims[1] * fDims[2]);
    for (int i = 0; i < fDims[0]; i++) {
        for (int j = 0; j < fDims[1]; j++) {
            for (int k = 0; k < fDims[2]; k++) {
                int irl = i + j * fDims[0] + k * fDims[1] * fDims[0];
                int ivoxel[3] = { i, j, k };
                fMedIndices[irl] = GetMaterialIndexOriginal(ivoxel);
            }
        }
    }
}


void Geom::Write(const std::string& fname, int nprimaries) {
    float fLBox = 1.f;
	std::vector<float> fEdepHist(fDims[2], 0.0f);
	for (int i = 0; i < fDims[0]; i++) {
		for (int j = 0; j < fDims[1]; j++) {
			for (int k = 0; k < fDims[2]; k++) {
				int index = i + j * fDims[0] + k * fDims[1] * fDims[0];
				fEdepHist[k] += fEndep[index];
			}
		}
	}

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
    fprintf(f, " %10d    %13.4e    %13.4e\n", i, (i+0.5f)*fLBox*toCm, fEdepHist[i]*norm/matDensity);
  }
  fclose(f);
  std::cout << " === Energy deposit histogram is written to the file:  " << fname << "\n" << std::endl;

  //writeMetaImage(fname + "dose", fDims, fSpacing, endep.data());
}
