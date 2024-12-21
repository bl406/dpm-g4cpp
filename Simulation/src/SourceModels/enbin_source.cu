#include "enbin_source.h"
#include "Track.hh"
#include <iostream>
#include "Utils.h"

namespace EnergyBinSource {

    __constant__ size_t d_nparticle;
    __constant__ size_t d_ngenerated;   
	__constant__ size_t d_nbins;
	cudaArray arrEnBins, arrEnPdf;
	cudaTextureObject_t texEnBins,texEnPdf;
	__device__ cudaTextureObject_t d_texEnBins, d_texEnPdf;

    __device__ float getEnergy(int i)
    {
        int ip = d_ngenerated + i;
        double ipp = (double)ip / d_nparticle;

        int energy_bin = 0;
        float cum_pdf = 0;
        for (int j = d_nbins - 1; j >= 0; --j) {
            cum_pdf += tex1D<float>(d_texEnPdf, j+0.5f);
            if (ipp < cum_pdf) {
                energy_bin = j;
                break;
            }
        }
		return tex1D<float>(d_texEnBins, energy_bin + 0.5f);	
    }
}

void EnbinSource::Initialize() {
	int nbin = (int)en_bins.size();
	
	cudaMemcpyToSymbol(EnergyBinSource::d_nparticle, &nparticle, sizeof(size_t));
	cudaMemcpyToSymbol(EnergyBinSource::d_ngenerated, &ngenerated, sizeof(size_t));

	cudaTextureDesc  texDesc;	
	memset(&texDesc, 0, sizeof(cudaTextureDesc));
	texDesc.normalizedCoords = 0;
	texDesc.filterMode = cudaFilterModePoint;
	texDesc.addressMode[0] = cudaAddressModeClamp;

	initCudaTexture(en_bins.data(), &nbin, 1, texDesc, EnergyBinSource::texEnBins, EnergyBinSource::arrEnBins);
	initCudaTexture(en_pdf.data(), &nbin, 1, texDesc, EnergyBinSource::texEnPdf, EnergyBinSource::arrEnPdf);
	cudaMemcpyToSymbol(EnergyBinSource::d_texEnBins, &EnergyBinSource::texEnBins, sizeof(cudaTextureObject_t));
	cudaMemcpyToSymbol(EnergyBinSource::d_texEnPdf, &EnergyBinSource::texEnPdf, sizeof(cudaTextureObject_t));

	cudaMemcpyToSymbol(EnergyBinSource::d_nbins, &nbins, sizeof(size_t));
}

void EnbinSource::InitTracks(Track* track, int n) {

//#pragma omp parallel for 
	for (int i = 0; i < n; ++i) {
		float x, y, z, u, v, w, wt, e, dnear;
		int ix, iy, iz, iq;

		int ip = ngenerated + i;
		double ipp = (double)ip / nparticle;

		int energy_bin = 0;
		float cum_pdf = 0;
		for (int j = en_pdf.size()-1; j >= 0; --j) {
			cum_pdf += en_pdf[j];
			if (ipp < cum_pdf) {
				energy_bin = j;
				break;
			}
		}

		setHistory(iq, e, x, y, z, ix, iy, iz, u, v, w, wt, dnear);

		track[i].fEkin = en_bins[energy_bin];
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

	ngenerated += n;
}