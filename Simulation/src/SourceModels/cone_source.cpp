#include "cone_source.h"
#include <string>
#include <math.h>
#include <cfloat>

#include "random.hh"
#include "omc_utilities.h"
#include "Utils.h"

namespace ConeSrc {	
	__constant__ float d_sad, d_spd;
	__constant__ float d_xiso, d_yiso, d_ziso;
	__constant__ float d_phi, d_theta, d_phicol;
	
    cudaArray arrR, arrRP;
	cudaTextureObject_t texR, texRP;
	__constant__ cudaTextureObject_t d_texR, d_texRP;
    __constant__ int d_nbins;

    __device__ int rad_idx_old;
    __device__ float f_old;

    __device__ void setHistory(
        int& iq, float& e,
        float& x, float& y, float& z,
        int& ix, int& iy, int& iz,
        float& u, float& v, float& w, float& wt,
        float& dnear) {

        iq = 0;
        wt = 1;
       
        // 随机读取一个radius
        int rad_idx;

        while (1) {
            rad_idx = int(d_nbins * CuRand::rand());
            if (rad_idx > d_nbins - 1) {
                rad_idx = d_nbins - 1;
            }

			float f = tex1D<float>(d_texRP, rad_idx);
            if (f < 1.e-6f) continue;

            if (CuRand::rand() < f / (f_old + FLT_EPSILON)) {
                rad_idx_old = rad_idx;
                f_old = f;
            }
            break;
        }

        // 随机读取一个theta
        float polar_theta = CuRand::rand() * 2 * M_PI;

        float radi;
        if (rad_idx_old == 0) {
            radi = tex1D<float>(d_texR, rad_idx_old) * CuRand::rand();
        }
        else {
			radi = tex1D<float>(d_texR, rad_idx_old - 1) + (tex1D<float>(d_texR, rad_idx_old) - tex1D<float>(d_texR, rad_idx_old - 1)) * CuRand::rand();            
        }
        // radi = sad / spd * radi;
        // radi = r[rad_idx - 1 ] + (r[rad_idx] - r[rad_idx - 1]) * CuRand::rand();
        x = radi * cos(polar_theta);
        y = radi * sin(polar_theta);
        z = d_spd;

        u = x;
        v = y;
        w = d_spd;

        // normalise direction
        float norm = sqrt(u * u + v * v + w * w);
        u /= norm;
        v /= norm;
        w /= norm;
    
        // the beam source is at (0, 0, -sad) in the phantom coordinate system and 
        // its axis aligh with the axis of the phantom coordinate system
        z -= d_sad; // tranform the position from beam coordinate system to phantom coordinate system

        cal_entrance_point(d_phi, d_theta, d_phicol, d_xiso, d_yiso, d_ziso, x, y, z, ix, iy, iz, u, v, w);

        dnear = 0.;
    }

    __device__ void InitTracksKernel(Track* track, int n) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i >= n) return;

        float x, y, z, u, v, w, wt, e, dnear;
        int ix, iy, iz, iq;

        setHistory(iq, e, x, y, z, ix, iy, iz, u, v, w, wt, dnear);

        track[i].fEkin = d_en_bins[energy_bin];
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

ConeSource::ConeSource(Geom* geom) : EnbinSource(geom) {
    char buffer[1024];
    printf("ConeSource::ConeSource() - Initializing cone source...\n");

    /* Read source xiso(cm) */
    if (getInputValue(buffer, "xiso") != 1) {
        printf("Can not find 'xiso' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    xiso = (float)atof(buffer);

    /* Read source yiso(cm) */
    if (getInputValue(buffer, "yiso") != 1) {
        printf("Can not find 'yiso' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    yiso = (float)atof(buffer);

    /* Read source ziso(cm) */
    if (getInputValue(buffer, "ziso") != 1) {
        printf("Can not find 'ziso' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    ziso = (float)atof(buffer);

    /* Read source theta */
    if (getInputValue(buffer, "theta") != 1) {
        printf("Can not find 'theta' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    theta = (float)atof(buffer);

    /* Read source phi */
    if (getInputValue(buffer, "phi") != 1) {
        printf("Can not find 'phi' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    phi = (float)atof(buffer);

    /* Read source phicol */
    if (getInputValue(buffer, "phicol") != 1) {
        printf("Can not find 'phicol' key on input file.\n");
        exit(EXIT_FAILURE);
    }
    phicol = (float)atof(buffer);

    if (getInputValue(buffer, "sad") != 1) {
        // throw std::exception("Required parameter not found");
        throw std::invalid_argument("Required parameter not found");
    }
    sad = (float)atof(buffer);
    if (getInputValue(buffer, "spd") != 1) {
        // throw std::exception("Required parameter not found");
        throw std::invalid_argument("Required parameter not found");
    }
    spd = (float)atof(buffer);
    if (getInputValue(buffer, "radius") != 1) {
        // throw std::exception("Required parameter not found");
        throw std::invalid_argument("Required parameter not found");
    }
    radius = (float)atof(buffer);

    // 读取flumap
    // if (getInputValue(buffer, "flumap") != 1) {
    //     // throw std::exception("Required parameter not found");
    //     throw std::invalid_argument("Required parameter not found");
    // }
    // removeSpaces(buffer, buffer);
    // FILE* fp = fopen(buffer, "rb");
    // if (!fp) {
    //     // throw std::exception("Could not open file");
    //     throw std::invalid_argument("Required parameter not found");
    // }
    // flumap.resize(nx * ny);
    // fread(&flumap[0], sizeof(float), nx * ny, fp);
    // fclose(fp);

    // 读取profile
    if (getInputValue(buffer, "Profile") != 1) {
        // throw std::exception("Required parameter not found");
        throw std::invalid_argument("Required parameter not found");
    }
    removeSpaces(buffer, buffer);
    FILE* fp = fopen(buffer, "rb");
    if (!fp) {
        // throw std::exception("Could not open file");
        throw std::invalid_argument("Required parameter not found");
    }
    float val1, val2, val3; // val1:r, val2: p_r, val3为误差
    while (fscanf(fp, "%f %f %f", &val1, &val2, &val3) == 3) {
        r.push_back(val1);
        p_r.push_back(val2);
    }
    printf("r:");
    for (auto it = r.begin(); it != r.end(); ++it) {
        printf("%f \t", *it);
    }
    printf("\n p_r:");
    for (auto it = p_r.begin(); it != p_r.end(); ++it) {
        printf("%f \t", *it);
    }
    printf("\n");


    // 读取能谱文件
    if (getInputValue(buffer, "spectrum file") != 1) {
        // throw std::exception("Required parameter not found");
        throw std::invalid_argument("Required parameter not found");
    }
    removeSpaces(buffer, buffer);
    if ((fp = fopen(buffer, "r")) == NULL) {
        printf("Unable to open file: %s\n", buffer);
        // throw std::exception("Unable to open file");
        throw std::invalid_argument("Unable to open file");
    }

    int ne;
    fgets(buffer, BUFFER_SIZE, fp);
    ne = atoi(buffer);
    fgets(buffer, BUFFER_SIZE, fp);
    en_bins = getValues<float>(buffer, ne);
    fgets(buffer, BUFFER_SIZE, fp);
    en_pdf = getValues<float>(buffer, ne);
    fclose(fp);

    // Print the energy bins and PDF
    printf("Energy bins: ");
    for (int i = 0; i < en_bins.size(); i++) {
        printf("%f ", en_bins[i]);
    }
    printf("\n");
    printf("Energy PDF: ");
    for (int i = 0; i < en_pdf.size(); i++) {
        printf("%f ", en_pdf[i]);
    }
    printf("\n");

    // initialize fluence sampling
    // ix_old = int(nx / 2);
    // iy_old = int(ny / 2);
    // f_old = flumap[ix_old + iy_old * nx];
    rad_idx_old = int(d_nbins / 2);
    f_old = p_r[rad_idx_old];
}

void ConeSource::initialize()
{
	EnbinSource::initialize();  // Must called to initialize the base class
	
	cudaMemcpyToSymbol(ConeSrc::d_sad, &sad, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_spd, &spd, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_xiso, &xiso, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_yiso, &yiso, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_ziso, &ziso, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_phi, &phi, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_theta, &theta, sizeof(float));
	cudaMemcpyToSymbol(ConeSrc::d_phicol, &phicol, sizeof(float));
	
    cudaTextureDesc  texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.normalizedCoords = 0;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.addressMode[0] = cudaAddressModeClamp;

	int nbin = r.size();
	cudaMemcpyToSymbol(ConeSrc::d_nbins, &nbin, sizeof(int));
    initCudaTexture(r.data(), &nbin, 1, texDesc, ConeSrc::texR, ConeSrc::arrR);
	initCudaTexture(p_r.data(), &nbin, 1, texDesc, ConeSrc::texRP, ConeSrc::arrRP);
	cudaMemcpyToSymbol(ConeSrc::d_texR, &ConeSrc::texR, sizeof(cudaTextureObject_t));
	cudaMemcpyToSymbol(ConeSrc::d_texRP, &ConeSrc::texRP, sizeof(cudaTextureObject_t));

    cudaMemcpy(&ConeSrc::rad_idx_old, &rad_idx_old, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(&ConeSrc::f_old, &f_old, sizeof(float), cudaMemcpyHostToDevice);
}

void ConeSource::setHistory(
    int& iq, float& e,
    float& x, float& y, float& z,
	int& ix, int& iy, int& iz,  
    float& u, float& v, float& w, float& wt,
    float& dnear) {
    iq = 0;
    wt = 1;
    float en_interval = en_bins[1] - en_bins[0];
    e = en_bins[energy_bin] - CuRand::rand() * en_interval;
    // 随机读取一个radius

    int rad_idx;

    while (1) {
        rad_idx = int(d_nbins * CuRand::rand());
        if (rad_idx > d_nbins - 1) {
            rad_idx = (int)d_nbins - 1;
        }

        float f = p_r[rad_idx];
        if (f < e - 6) continue;

        if (CuRand::rand() < f / (f_old + DBL_EPSILON)) {
            rad_idx_old = rad_idx;
            f_old = f;
        }

        break;
    }
    // 随机读取一个theta

    float polar_theta = CuRand::rand() * 2 * M_PI;

    float radi;
    if (rad_idx_old == 0) {
        radi = r[rad_idx_old] * CuRand::rand();
    }
    else {
        radi = r[rad_idx_old - 1] + (r[rad_idx_old] - r[rad_idx_old - 1]) * CuRand::rand();
    }
    // radi = sad / spd * radi;
    // radi = r[rad_idx - 1 ] + (r[rad_idx] - r[rad_idx - 1]) * CuRand::rand();
    x = radi * cos(polar_theta);
    y = radi * sin(polar_theta);
    z = spd;

    u = x;
    v = y;
    w = spd;

    // normalise direction
    float norm = sqrt(u * u + v * v + w * w);
    u /= norm;
    v /= norm;
    w /= norm;

    dnear = 0.;
    // the beam source is at (0, 0, -sad) in the phantom coordinate system and 
    // its axis aligh with the axis of the phantom coordinate system
    z -= sad; // tranform the position from beam coordinate system to phantom coordinate system
    // printf("x: %f, y: %f, z: %f, u: %f, v: %f, w: %f\n", x, y, z, u, v, w);
    cal_entrance_point(geometry, phi, theta, phicol, xiso, yiso, ziso, x, y, z, ix, iy, iz, u, v, w);
}

