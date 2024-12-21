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
    __constant__ float d_energy;
    __constant__ int d_srctype;
    __constant__ int d_deltak;
	__constant__ float d_cdfinv1[INVDIM];
    __constant__ float d_cdfinv2[INVDIM];
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
        float rnno1;
        float rnno2;

        iq = d_charge;

        /* Get primary particle energy */
        float ein = 0.0f;
        switch (d_srctype) {
        case CollipntSource::MONOENERGETIC:
            ein =d_energy;
            break;
        case CollipntSource::SPECTRUM:
            /* Sample initial energy from spectrum data */
            rnno1 = CuRand::rand();
            rnno2 = CuRand::rand();

            /* Sample bin number in order to select particle energy */
            int k = (int)fmin(d_deltak * rnno1, d_deltak - 1.0f);
            ein = d_cdfinv1[k] + rnno2 * d_cdfinv2[k];
            break;
            // default:
                // break;
        }

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

    /* Get source file path from input data */
    source_type = SPECTRUM;    /* energy spectrum as default case */

    /* First check of spectrum file was given as an input */
    if (getInputValue(buffer, "spectrum file") != 1) {
        printf("Can not find 'spectrum file' key on input file.\n");
        printf("Switch to monoenergetic case.\n");
        source_type = MONOENERGETIC;    /* monoenergetic source */
    }

    switch (source_type) {
    case MONOENERGETIC:
        if (getInputValue(buffer, "mono energy") != 1) {
            printf("Can not find 'mono energy' key on input file.\n");
            exit(EXIT_FAILURE);
        }
        energy = (float)atof(buffer);
        printf("%f monoenergetic source\n", energy);
        break;

    case SPECTRUM:
        removeSpaces(spectrum_file, buffer);

        /* Open .source file */
        FILE* fp;

        if ((fp = fopen(spectrum_file, "r")) == NULL) {
            printf("Unable to open file: %s\n", spectrum_file);
            exit(EXIT_FAILURE);
        }

        printf("Path to spectrum file : %s\n", spectrum_file);

        /* Read spectrum file title */
        fgets(buffer, BUFFER_SIZE, fp);
        printf("Spectrum file title: %s", buffer);

        /* Read number of bins and spectrum type */
        float enmin;   /* lower energy of first bin */
        int nensrc;     /* number of energy bins in spectrum histogram */
        int imode;      /* 0 : histogram counts/bin, 1 : counts/MeV*/

        fgets(buffer, BUFFER_SIZE, fp);
        sscanf(buffer, "%d %f %d", &nensrc, &enmin, &imode);

        if (nensrc > MXEBIN) {
            printf("Number of energy bins = %d is greater than max allowed = "
                "%d. Increase MXEBIN macro!\n", nensrc, MXEBIN);
            exit(EXIT_FAILURE);
        }

        /* upper energy of bin i in MeV */
        float* ensrcd = (float*)malloc(nensrc * sizeof(float));
        /* prob. of finding a particle in bin i */
        float* srcpdf = (float*)malloc(nensrc * sizeof(float));

        /* Read spectrum information */
        for (int i = 0; i < nensrc; i++) {
            fgets(buffer, BUFFER_SIZE, fp);
            sscanf(buffer, "%f %f", &ensrcd[i], &srcpdf[i]);
        }
        printf("Have read %d input energy bins from spectrum file.\n", nensrc);

        if (imode == 0) {
            printf("Counts/bin assumed.\n");
        }
        else if (imode == 1) {
            printf("Counts/MeV assumed.\n");
            srcpdf[0] *= (ensrcd[0] - enmin);
            for (int i = 1; i < nensrc; i++) {
                srcpdf[i] *= (ensrcd[i] - ensrcd[i - 1]);
            }
        }
        else {
            printf("Invalid mode number in spectrum file.");
            exit(EXIT_FAILURE);
        }

        float ein = ensrcd[nensrc - 1];
        printf("Energy ranges from %f to %f MeV\n", enmin, ein);

        /* Initialization routine to calculate the inverse of the
        cumulative probability distribution that is used during execution to
        sample the incident particle energy. */
        float* srccdf = (float*)malloc(nensrc * sizeof(float));

        srccdf[0] = srcpdf[0];
        for (int i = 1; i < nensrc; i++) {
            srccdf[i] = srccdf[i - 1] + srcpdf[i];
        }

        float fnorm = 1.0f / srccdf[nensrc - 1];
        int binsok = 0;
        deltak = INVDIM; /* number of elements in inverse CDF */
        float gridsz = 1.0f / deltak;

        for (int i = 0; i < nensrc; i++) {
            srccdf[i] *= fnorm;
            if (i == 0) {
                if (srccdf[0] <= 3.0f * gridsz) {
                    binsok = 1;
                }
            }
            else {
                if ((srccdf[i] - srccdf[i - 1]) < 3.0f * gridsz) {
                    binsok = 1;
                }
            }
        }

        if (binsok != 0) {
            printf("Warning!, some of normalized bin probabilities are "
                "so small that bins may be missed.\n");
        }

        /* Calculate cdfinv. This array allows the rapid sampling for the
        energy by precomputing the results for a fine grid. */
        cdfinv1 = (float*)malloc(size_t(deltak * sizeof(float)));
        cdfinv2 = (float*)malloc(size_t(deltak * sizeof(float)));
        float ak;

        for (int k = 0; k < deltak; k++) {
            ak = (float)k * gridsz;
            int i;

            for (i = 0; i < nensrc; i++) {
                if (ak <= srccdf[i]) {
                    break;
                }
            }

            /* We should fall here only through the above break sentence. */
            if (i != 0) {
                cdfinv1[k] = ensrcd[i - 1];
            }
            else {
                cdfinv1[k] = enmin;
            }
            cdfinv2[k] = ensrcd[i] - cdfinv1[k];
        }

        /* Cleaning */
        fclose(fp);
        free(ensrcd);
        free(srcpdf);
        free(srccdf);

        break;

        // default:
        //     printf("Error\n");
        //     exit(EXIT_FAILURE);
    }

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
    int ndeltak = (int)deltak;
    cudaMemcpyToSymbol(CollipntSrc::d_charge, &charge, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_energy, &energy, sizeof(float));
    cudaMemcpyToSymbol(CollipntSrc::d_srctype, &source_type, sizeof(int));
    cudaMemcpyToSymbol(CollipntSrc::d_deltak, &ndeltak, sizeof(int));
    cudaMemcpyToSymbol(CollipntSrc::d_cdfinv1, cdfinv1, sizeof(float) * ndeltak);
    cudaMemcpyToSymbol(CollipntSrc::d_cdfinv2, cdfinv2, sizeof(float) * ndeltak);
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

    CudaCheckError();
}

