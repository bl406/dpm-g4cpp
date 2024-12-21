#pragma once

#include "enbin_source.h"
#include <vector>

class Geom;

class ConeSource : public EnbinSource {
public:
	ConeSource(Geom* geom);
	~ConeSource() {};

	// setHistory
	void setHistory(
		int& iq, float& e,
		float& x, float& y, float& z,
		int& ix, int& iy, int& iz,
		float& u, float& v, float& w, float& wt,
		float& dnear
	);

	float normalized_factor() {
		return 1.0;
	}

	void Initialize() override;

protected:
	float xiso;
	float yiso;
	float ziso;
	float theta;
	float phi;
	float phicol;

	std::vector<float> r;
	std::vector<float> p_r;
	float sad;
	float spd; // Source to scoring plane

	float radius; // cone radius
	int rad_idx_old;
	float f_old;
};