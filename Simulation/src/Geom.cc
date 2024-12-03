
#include "Geom.hh"

#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>

#include "metaio.h"

int _GetMaterialIndex(int fPreDefinedGeomIndex, double fInvLBox, int* i) {
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


void Geom::InitGeom(int geomIndex){
	spacing.fill(1.0);
	inv_spacing.fill(1.0);
    switch (geomIndex)
    {
	case 3:
    default:
		xbound = { -5.5, 5.5 };
		ybound = { -5.5, 5.5 };
		zbound = { -0.5, 24.5 };
        break;;
    }

    dims[0] = static_cast<int>((xbound[1] - xbound[0]) / spacing[0]);
    dims[1] = static_cast<int>((ybound[1] - ybound[0]) / spacing[1]);
    dims[2] = static_cast<int>((zbound[1] - zbound[0]) / spacing[2]);

	med_indices.resize(dims[0] * dims[1] * dims[2]);
	for (int i = 0; i < dims[0]; i++) {
		for (int j = 0; j < dims[1]; j++) {
			for (int k = 0; k < dims[2]; k++) {
				int irl = i + j * dims[0] + k * dims[1] * dims[0];
				int ivoxel[3] = { i, j, k };
				med_indices[irl] = _GetMaterialIndex(geomIndex, inv_spacing[2], ivoxel);
			}
		}
	}
}

bool Geom::LoadEgsPhantom(const std::string& fname){
	const int BUFFER_SIZE = 256;
	char buffer[BUFFER_SIZE];

    /* Open .egsphant file */
    FILE* fp;

    if ((fp = fopen(fname.c_str(), "r")) == NULL) {
        printf("Unable to open file: %s\n", fname);
        return false;
    }

    printf("Path to phantom file : %s\n", fname);

    /* Get number of media in the phantom */
    fgets(buffer, BUFFER_SIZE, fp);
    int nmed = atoi(buffer);
	std::vector<std::string> med_names(nmed);
    /* Get media names on phantom file */
    for (int i = 0; i < nmed; i++) {
        fgets(buffer, BUFFER_SIZE, fp);     
		med_names[i] = buffer;
    }
    if (nmed != fMaterialData->fNumMaterial) {
		printf("Number of media in phantom file does not match with the number of materials in the material file\n");
		return false;
    }

	std::map<int, int> med_map;
	for (int i = 0; i < nmed; i++) {
		med_map[i] = std::find(fMaterialData->fMaterialName.begin(), fMaterialData->fMaterialName.end(), med_names[i]) 
            - fMaterialData->fMaterialName.begin();
		if (med_map[i] == -1) {
			printf("Material %s not found in the material file\n", med_names[i].c_str());
			return false;
		}
	}

    /* Skip next line, it contains dummy input */
    fgets(buffer, BUFFER_SIZE, fp);

    /* Read voxel numbers on each direction */
    fgets(buffer, BUFFER_SIZE, fp);
    sscanf(buffer, "%d %d %d", &dims[0], &dims[1], &dims[2]);

    std::vector<float> xbounds, ybounds, zbounds;
    std::vector<float> med_densities;
    /* Read voxel boundaries on each direction */
	xbounds.resize(dims[0]);
	for (int i = 0; i < dims[0]; i++) {
		fscanf(fp, "%f", &xbounds[i]);
	}
	ybounds.resize(dims[1]);
	for (int i = 0; i < dims[1]; i++) {
		fscanf(fp, "%f", &ybounds[i]);
	}
	zbounds.resize(dims[2]);
	for (int i = 0; i < dims[2]; i++) {
		fscanf(fp, "%f", &zbounds[i]);
	}

	zbound[0] = xbounds[0];    
	zbound[1] = xbounds[dims[0] - 1];
	ybound[0] = ybounds[0];
	ybound[1] = ybounds[dims[1] - 1];
	zbound[0] = zbounds[0];
	zbound[1] = zbounds[dims[2] - 1];

    /* Skip the rest of the last line read before */
    fgets(buffer, BUFFER_SIZE, fp);

    /* Read media indices */
    int irl = 0;    // region index
    char idx;
	med_indices.resize(dims[0] * dims[1] * dims[2]);

    for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int i = 0; i < dims[0]; i++) {
                irl = i + j * dims[0] + k * dims[1] * dims[0];
                idx = fgetc(fp);
                /* Convert digit stored as char to int */
                med_indices[irl] = idx - '0';
				med_indices[irl] = med_map[med_indices[irl]];
            }
            /* Jump to next line */
            fgets(buffer, BUFFER_SIZE, fp);
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }

    /* Read media densities */
	med_densities.resize(dims[0] * dims[1] * dims[2]);
    for (int k = 0; k < dims[2]; k++) {
        for (int j = 0; j < dims[1]; j++) {
            for (int i = 0; i < dims[0]; i++) {
                irl = i + j * dims[0] + k * dims[1] * dims[0];
                fscanf(fp, "%f", &med_densities[irl]);
            }
        }
        /* Skip blank line */
        fgets(buffer, BUFFER_SIZE, fp);
    }

	spacing[0] = (xbounds[dims[0] - 1] - xbounds[0]) / dims[0];
	spacing[1] = (ybounds[dims[1] - 1] - ybounds[0]) / dims[1];
	spacing[2] = (zbounds[dims[2] - 1] - zbounds[0]) / dims[2];

	inv_spacing[0] = 1.0 / spacing[0];
	inv_spacing[1] = 1.0 / spacing[1];
	inv_spacing[2] = 1.0 / spacing[2];

    

    /* Summary with geometry information */
    printf("Number of media in phantom : %d\n", nmed);
    printf("Media names: ");
    for (int i = 0; i < nmed; i++) {
        printf("%s, ", med_names[i]);
    }
    printf("\n");
    printf("Number of voxels on each direction (X,Y,Z) : (%d, %d, %d)\n",
        dims[0], dims[1], dims[2]);
    printf("Minimum and maximum boundaries on each direction : \n");
    printf("\tX (cm) : %lf, %lf\n",
       xbounds[0],xbounds[dims[0]]);
    printf("\tY (cm) : %lf, %lf\n",
       ybounds[0],ybounds[dims[1]]);
    printf("\tZ (cm) : %lf, %lf\n",
       zbounds[0],zbounds[dims[2]]);

    /* Close phantom file */
    fclose(fp);

    return true;
}

float Geom::DistanceToBoundary(double* r, double* v, int* i) {
    static float kHalfTolerance = 0.5 * kTolerance;

    if(r[0] < xbound[0] || r[0] > xbound[1]
        || r[1] < ybound[0] || r[1] > ybound[1]
		|| r[2] < zbound[0] || r[2] > zbound[1]) 
    {
			return -1.0;
	}
  
    // compute the current x,y and z box/volume index based on the current r(rx,ry,rz)
    // round downward to integer
    i[0] = std::floor((r[0] - xbound[0]) * inv_spacing[0]);
    i[1] = std::floor((r[1] - ybound[0]) * inv_spacing[1]);
    i[2] = std::floor((r[2] - zbound[0]) * inv_spacing[2]);
    //
    // transform the r(rx,ry,rz) into the current local box
    const float trX = r[0] - xbound[0] - i[0] * spacing[0] - 0.5 * spacing[0];
    const float trY = r[1] - ybound[0] - i[1] * spacing[1] - 0.5 * spacing[1];;
    const float trZ = r[2] - zbound[0] - i[2] * spacing[2] - 0.5 * spacing[2];;
    //
    // compute the distance to boundary in the local box (centered around 0,0,0)
    float pdist = 0.0;
    float stmp = 0.0;
    float snext = 1.0E+20;
    //
    // calculate x
    if (v[0] > 0.0) {
        pdist = spacing[0] * 0.5f - trX;
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
        pdist = spacing[0] * 0.5f + trX;
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
        pdist = spacing[1] * 0.5f - trY;
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
        pdist = spacing[1] * 0.5f + trY;
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
        pdist = spacing[2] * 0.5f - trZ;
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
        pdist = spacing[2] * 0.5f + trZ;
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

void Geom::Score(double edep, int* ivoxel){
	int index = ivoxel[0] + ivoxel[1] * dims[0] + ivoxel[2] * dims[1] * dims[0];
	endep[index] += edep;
}

void Geom::InitScore()
{
	ensrc = 0.0;
	endep.resize(dims[0] * dims[1] * dims[2], 0.f);
	accum_endep.resize(dims[0] * dims[1] * dims[2], 0.f);
	accum_endep2.resize(dims[0] * dims[1] * dims[2], 0.f);
}

void Geom::Write(const std::string& fname, int nprimaries) {

	writeMetaImage(fname+"dose", dims, spacing, endep.data());
}