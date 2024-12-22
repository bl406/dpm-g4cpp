#pragma once

#include "source.h"
#include "egs_alias_table.h"
#include <vector>

class Geom;

class ConeSource : public Source {
public:
	ConeSource(Geom* geom);
	~ConeSource();

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

	float sad;
	float spd; // Source to scoring plane

	EGS_AliasTable alias_table;
	float radius;
};