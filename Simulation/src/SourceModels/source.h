/**
 * @file source.h
 * @brief This file contains the declaration of the Source class.
 */

#ifndef SOURCE_H
#define SOURCE_H

#include <vector>
#include <string>
#include <cuda_runtime.h>

class Track;
class Geom;

/* Physical constants */
#ifndef RM
#define RM 0.5109989461f     // MeV * c^(-2)
#endif

const int MXEBIN = 200;     // number of energy bins of spectrum
const int INVDIM = 1000;    // number of bins in inverse CDF

const float M_PI = 3.14159265358979323846f;

extern int isource;
extern class Source* source;
Source* initSource(Geom* geom, std::string srcFn);
void cleanSource();

extern __device__ void cal_entrance_point(float phi, float theta, float phicol,
    float xiso, float yiso, float ziso,
    float& x, float& y, float& z, 
	int& ix, int& iy, int& iz,
    float& u, float& v, float& w);

namespace Spectrum {
	extern __device__ float getEkin();
}

class Source {
public:
	enum { MONOENERGETIC, SPECTRUM } source_type;
	enum SourceType { COLLIPOINT_SOURCE = 1, CONE_SOURCE };

	Source(Geom* geom);
	virtual ~Source();
    
    virtual void InitTracks(Track* track, int n) = 0;

    virtual void Initialize();

	virtual float normalized_factor() = 0;
protected:
    Geom* geometry;	

	/* For monoenergetic source */
	float energy;

	/* For spectrum */
	float deltak;              // number of elements in inverse CDF
	float* cdfinv1;            // energy value of bin
	float* cdfinv2;            // prob. that particle has energy xi
};


#endif // SOURCE_H