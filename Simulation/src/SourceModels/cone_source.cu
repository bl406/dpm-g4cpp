#include "cone_source.h"
#include <cuda_runtime.h>
#include <string>
#include <math.h>
#include <cfloat>

#include "random.hh"
#include "omc_utilities.h"
#include "Utils.h"
#include "Track.hh"

namespace ConeSrc {	
	__constant__ float d_sad, d_spd;
	__constant__ float d_xiso, d_yiso, d_ziso;
	__constant__ float d_phi, d_theta, d_phicol;
	
    cudaArray_t arrATXi, arrATWi, arrATBin;
    cudaTextureObject_t texATXi, texATWi, texATBin;
	__constant__ cudaTextureObject_t d_texATXi, d_texATWi, d_texATBin;
    __constant__ int d_ATNp, d_ATType;

    inline __device__ float getRadius() {
		float r1 = CuRand::rand();
        float aj = r1 * d_ATNp;
        int j = (int)aj;
        aj -= j;
        if (aj > tex1D<float>(d_texATWi, j+0.5f)) {
            j = tex1D<float>(d_texATBin, j + 0.5f);
        }
        if (!d_ATType) {
            return tex1D<float>(d_texATXi, j + 0.5f);
        }
        float x = tex1D<float>(d_texATXi, j + 0.5f);
        float dx = tex1D<float>(d_texATXi, j +1 + 0.5f) - x;
        float r2 = CuRand::rand();

        return x + dx * r2;
    }

    __device__ void setHistory(
        int& iq, float& e,
        float& x, float& y, float& z,
        int& ix, int& iy, int& iz,
        float& u, float& v, float& w, float& wt,
        float& dnear) {

        iq = 0;
        wt = 1;
 
        e = Spectrum::getEkin();

        // 随机读取一个theta
        float polar_theta = CuRand::rand() * 2 * 3.14159265358979323846f;
        float radi = getRadius();
       
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

ConeSource::ConeSource(Geom* geom) : Source(geom) {
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
        throw std::invalid_argument("Required parameter not found");
    }
    spd = (float)atof(buffer);
    if (getInputValue(buffer, "radius") != 1) {
        throw std::invalid_argument("Required parameter not found");
    }
    radius = (float)atof(buffer);

    // 读取profile
    if (getInputValue(buffer, "Profile") != 1) {
        throw std::invalid_argument("Required parameter not found");
    }
    removeSpaces(buffer, buffer);
    FILE* fp = fopen(buffer, "rb");
    if (!fp) {
        throw std::invalid_argument("Required parameter not found");
    }

	std::vector<double> r, p_r;
    double val1, val2, val3; // val1:r, val2: p_r, val3为误差
    while (fscanf(fp, "%lf %lf %lf", &val1, &val2, &val3) == 3) {
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

	alias_table.initialize((int)p_r.size(), r.data(), p_r.data(), 1);
}

ConeSource::~ConeSource() {
   
}

void ConeSource::Initialize()
{
	Source::Initialize();  // Must called to initialize the base class
	
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

	int np = alias_table.getNp();
	cudaMemcpyToSymbol(ConeSrc::d_ATNp, &np, sizeof(int));
    std::vector<float> temp(np);
	double* ATXi = alias_table.getXi();
	for (int i = 0; i < np; i++) {
		temp[i] = (float)ATXi[i];
	}
	initCudaTexture(temp.data(), &np, 1, &texDesc, ConeSrc::texATXi, ConeSrc::arrATXi);
    double* ATWi = alias_table.getWi();
    for (int i = 0; i < np; i++) {
        temp[i] = (float)ATWi[i];
    }
    initCudaTexture(temp.data(), &np, 1, &texDesc, ConeSrc::texATWi, ConeSrc::arrATWi);
    initCudaTexture(alias_table.getBin(), &np, 1, &texDesc, ConeSrc::texATBin, ConeSrc::arrATBin);
    cudaMemcpyToSymbol(ConeSrc::d_texATXi, &ConeSrc::texATXi, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(ConeSrc::d_texATWi, &ConeSrc::texATWi, sizeof(cudaTextureObject_t));
    cudaMemcpyToSymbol(ConeSrc::d_texATBin, &ConeSrc::texATBin, sizeof(cudaTextureObject_t));

    CudaCheckError();
}
