#pragma once

#include "source.h"

class Geom;

namespace EnergyBinSource {
	__device__ float getEnergy(int i);
}

class EnbinSource : public Source {
public:
	EnbinSource(Geom* geom) : Source(geom), ngenerated(0){};
	~EnbinSource() {};

	std::vector<float> getEnBins() {
		return en_bins;
	}
	std::vector<float> getEnPdf() {
		return en_pdf;
	}

	int getEnergyBin() {
		return energy_bin;
	}
	void setEnergyBin(int bin) {
		energy_bin = bin;
	}

	void InitTracks(Track* track, int n) override;

	void setNParticle(size_t n) { nparticle = n; }

	void Initialize() override;
protected:
	size_t nparticle;
	int energy_bin;
	size_t ngenerated;
	std::vector<float> en_bins;	// Energy bins
	std::vector<float> en_pdf;	// Energy CDF
};