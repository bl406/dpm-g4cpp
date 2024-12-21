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

enum SourceType {
    COLLIPOINT_SOURCE = 1,
	CONE_SOURCE
};

class Source {
public:
	Source(Geom* geom) : geometry(geom){}
    virtual ~Source() {}
    
    virtual void InitTracks(Track* track, int n) = 0;

    virtual void Initialize() = 0;

	virtual float normalized_factor() = 0;

protected:
    Geom* geometry;	
};


#endif // SOURCE_H