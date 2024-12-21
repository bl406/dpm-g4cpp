#pragma once

#include "source.h"

class Geom;

/* ISOURCE = 3: Point source from the front with rectangluar collimation */
class CollipntSource : public Source {
public:
    enum { MONOENERGETIC, SPECTRUM } source_type;

    CollipntSource(Geom* geom);

    ~CollipntSource();

	void InitTracks(Track* d_track, int n) override;

    float normalized_factor() override{
        return xsize * ysize;
    }

	void Initialize() override; 

private:
    
    char spectrum_file[128];    // spectrum file path

    int charge;                 // 0 : photons, -1 : electron, +1 : positron

    /* For monoenergetic source */
    float energy;

    /* For spectrum */
    float deltak;              // number of elements in inverse CDF
    float* cdfinv1;            // energy value of bin
    float* cdfinv2;            // prob. that particle has energy xi

    /* Source shape information */
    float ssd;                 // distance of point source to phantom surface
    float xinl, xinu;          // lower and upper x-bounds of the field on
    // phantom surface
    float yinl, yinu;          // lower and upper y-bounds of the field on
    // phantom surface
    float xsize, ysize;        // x- and y-width of collimated field
    int ixinl, ixinu;        // lower and upper x-bounds indices of the
    // field on phantom surface
    int iyinl, iyinu;        // lower and upper y-bounds indices of the
    // field on phantom surface
};

