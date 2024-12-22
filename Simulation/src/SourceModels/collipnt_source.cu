#include "collipnt_source.h"
#include "math.h"
#include "omc_utilities.h"
#include "Random.hh"
#include "Geom.hh"
#include "track.hh"
#include "error_checking.h"
#include "Utils.h"

namespace CollipntSrc {

    __constant__ float d_charge;    
	__constant__ float d_xinl;
	__constant__ float d_xinu;
	__constant__ float d_yinl;
	__constant__ float d_yinu;
	__constant__ float d_xsize;
	__constant__ float d_ysize;
	__constant__ float d_ssd;
	__constant__ int d_ixinl;
	__constant__ int d_ixinu;
	__constant__ int d_iyinl;
	__constant__ int d_iyinu;

    __device__ void setHistory(int& iq, float& e, float& x, float& y, float& z, int& ix, int& iy, int& iz, float& u, float& v, float& w, float& wt, float& dnear) {

        iq = d_charge;

        float ein = Spectrum::getEkin();

        /* Check if the particle is an electron, in such a case add electron
        rest mass energy */
        if (iq != 0) {
            /* Electron or positron */
            e = ein + RM;
        }
        else {
            /* Photon */
            e = ein;
        }

        /* Set particle position. First obtain a random position in the rectangle
        defined by the collimator */
        float rxyz = 0.0;
        if (d_xsize == 0.0 || d_ysize == 0.0) {
            x = d_xinl;
            y = d_yinl;

            rxyz = sqrt(pow(d_ssd, 2.0f) + pow(x, 2.0f) +
                pow(y, 2.0f));

            /* Get direction along z-axis */
            w = d_ssd / rxyz;

        }
        else {
            float fw;
            float rnno3;
            do { /* rejection sampling of the initial position */
                rnno3 = CuRand::rand();
                x = rnno3 * d_xsize + d_xinl;
                rnno3 = CuRand::rand();
                y = rnno3 * d_ysize + d_yinl;
                rnno3 = CuRand::rand();
                rxyz = sqrt(d_ssd * d_ssd +
                    x * x +
                    y * y);

                /* Get direction along z-axis */
                w = d_ssd / rxyz;
                fw = w * w * w;
            } while (rnno3 >= fw);
        }
        /* Set position of the particle in front of the geometry */
		z = Geometry::d_Zbound[0];

        /* At this point the position has been found, calculate particle
        direction */
        u = x / rxyz;
        v = y / rxyz;

        /* Determine region index of source particle */
        if (d_xsize == 0.0f) {
            ix = d_ixinl;
        }
        else {
            ix = int((x - Geometry::d_Xbound[0]) / Geometry::d_Spacing[0]);
			if (ix > Geometry::d_Dim[0] - 1) ix = Geometry::d_Dim[0] - 1;		
        }
        if (d_ysize == 0.0f) {
            iy = d_iyinl;
        }
        else {           
            iy = int((y - Geometry::d_Ybound[0]) / Geometry::d_Spacing[1]);
            if (iy > Geometry::d_Dim[1] - 1) iy = Geometry::d_Dim[1] - 1;			
        }
        iz = 0;

        /* Set statistical weight and distance to closest boundary*/
        wt = 1.0f;
        dnear = 0.0f;
    }

    __global__ void InitTracksKernel(Track* track, int n) {
		int i = threadIdx.x + blockIdx.x * blockDim.x;
		if (i >= n) return;

        float x, y, z, u, v, w, wt, e, dnear;
        int ix, iy, iz, iq;
		x = y = z = u = v = w = wt = e = dnear = 0;
		ix = iy = iz = iq = 0;

		setHistory(iq, e, x, y, z, ix, iy, iz, u, v, w, wt, dnear);
        
		track[i].fEkin = e;
		track[i].fType = iq;
		track[i].fDirection[0] = u;
		track[i].fDirection[1] = v;
		track[i].fDirection[2] = w;
		track[i].fPosition[0] = x;
		track[i].fPosition[1] = y;
		track[i].fPosition[2] = z;
		track[i].fBoxIndx[0] = ix;
		track[i].fBoxIndx[1] = iy;
		track[i].fBoxIndx[2] = iz;
		track[i].fTrackLength = 0;
		track[i].fEdep = 0; 
    }
}

void CollipntSource::InitTracks(Track* d_track, int n) {

    int nblocks = divUp(n, THREADS_PER_BLOCK);
    CollipntSrc::InitTracksKernel << <nblocks, THREADS_PER_BLOCK >> > (d_track, n);
    CudaCheckError();
}


/* ISOURCE = 3: Point source from the front with rectangluar collimation */
CollipntSource::CollipntSource(Geom* geom) : Source(geom)
{
    char buffer[BUFFER_SIZE];

    /* Initialize geometrical data of the source */

    /* Read collimator rectangle */
    if (getInputValue(buffer, "collimator bounds") != 1) {
        printf("Can not find 'collimator bounds' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    sscanf(buffer, "%f %f %f %f", &xinl,
        &xinu, &yinl, &yinu);

    /* Calculate x-direction input zones */
    if (xinl < geometry->GetXbound()[0]) {
        xinl = geometry->GetXbound()[0];
    }
    if (xinu <= xinl) {
        xinu = xinl;  /* default a pencil beam */
    }

    /* Check radiation field is not too big against the phantom */
    if (xinu > geometry->GetXbound()[1]) {
        xinu = geometry->GetXbound()[1];
    }
    if (xinl > geometry->GetXbound()[1]) {
        xinl = geometry->GetXbound()[1];
    }

    /* Now search for initial region x index range */
    printf("Index ranges for radiation field:\n");
    ixinl = 0;
    while ((geometry->GetXbounds()[ixinl] <= xinl) &&
        (geometry->GetXbounds()[ixinl + 1] < xinl)) {
        ixinl++;
    }

    ixinu = ixinl - 1;
    while ((geometry->GetXbounds()[ixinu] <= xinu) &&
        (geometry->GetXbounds()[ixinu + 1] < xinu)) {
        ixinu++;
    }
    printf("i index ranges over i = %d to %d\n", ixinl, ixinu);

    /* Calculate y-direction input zones */
    if (yinl < geometry->GetYbound()[0]) {
        yinl = geometry->GetYbound()[0];
    }
    if (yinu <= yinl) {
        yinu = yinl;  /* default a pencil beam */
    }

    /* Check radiation field is not too big against the phantom */
    if (yinu > geometry->GetYbound()[1]) {
        yinu = geometry->GetYbound()[1];
    }
    if (yinl > geometry->GetYbound()[1]) {
        yinl = geometry->GetYbound()[1];
    }

    /* Now search for initial region y index range */
    iyinl = 0;
    while ((geometry->GetYbounds()[iyinl] <= yinl) &&
        (geometry->GetYbounds()[iyinl + 1] < yinl)) {
        iyinl++;
    }
    iyinu = iyinl - 1;
    while ((geometry->GetYbounds()[iyinu] <= yinu) &&
        (geometry->GetYbounds()[iyinu + 1] < yinu)) {
        iyinu++;
    }
    printf("j index ranges over i = %d to %d\n", iyinl, iyinu);

    /* Calculate collimator sizes */
    xsize = xinu - xinl;
    ysize = yinu - yinl;

    /* Read source charge */
    if (getInputValue(buffer, "charge") != 1) {
        printf("Can not find 'charge' key on input file.\n");
        exit(EXIT_FAILURE);
    }

    charge = atoi(buffer);
    if (charge < -1 || charge > 1) {
        printf("Particle kind not recognized.\n");
        exit(EXIT_FAILURE);
    }

    /* Read source SSD */
    if (getInputValue(buffer, "ssd") != 1) {
        printf("Can not find 'ssd' key on input file.\n");
        exit(EXIT_FAILURE);
    }

    ssd = (float)atof(buffer);
    if (ssd < 0) {
        printf("SSD must be greater than zero.\n");
        exit(EXIT_FAILURE);
    }

    /* Print some information for debugging purposes */
    if (false) {
        printf("Source information :\n");
        printf("\t Charge = %d\n", charge);
        printf("\t SSD (cm) = %f\n", ssd);
        printf("Collimator :\n");
        printf("\t x (cm) : min = %f, max = %f\n", xinl, xinu);
        printf("\t y (cm) : min = %f, max = %f\n", yinl, yinu);
        printf("Sizes :\n");
        printf("\t x (cm) = %f, y (cm) = %f\n", xsize, ysize);
    }
}

CollipntSource::~CollipntSource()
{
}


void CollipntSource::Initialize() {
    cudaMemcpyToSymbol(CollipntSrc::d_charge, &charge, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_xinl, &xinl, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_xinu, &xinu, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_yinl, &yinl, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_yinu, &yinu, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_xsize, &xsize, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_ysize, &ysize, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_ssd, &ssd, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_ixinl, &ixinl, sizeof(int));
    cudaMemcpyToSymbol(CollipntSrc::d_ixinu, &ixinu, sizeof(int));
    cudaMemcpyToSymbol(CollipntSrc::d_iyinl, &iyinl, sizeof(int));
	cudaMemcpyToSymbol(CollipntSrc::d_iyinu, &iyinu, sizeof(int));

	Source::Initialize();

    CudaCheckError();
}

