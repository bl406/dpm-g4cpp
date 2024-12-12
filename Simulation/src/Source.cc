#include "Source.hh"
#include "Track.hh"
#include "Utils.h"

namespace SimpleSourceKernel {
	void InitTracksKernel(Track* track, int n, float ekin, int type, float lbox) {
		for(int index = 0; index < n; ++index)
		{
			track[index].fEkin = ekin;
			track[index].fType = type;
			track[index].fDirection[0] = 0.0f;                  // initial direction is [0,0,1]
			track[index].fDirection[1] = 0.0f;
			track[index].fDirection[2] = 1.0f;
			//
			track[index].fPosition[0] = 0.0f;                   // initial position is [0,0, theRZ0]
			track[index].fPosition[1] = 0.0f;
			track[index].fPosition[2] = -0.5f * lbox;

			track[index].fTrackLength = 0;
		}
	}
}

SimpleSource::SimpleSource(float ekin, int type, float lbox) {
	fEkin = ekin;
	fType = type;
	fLBox = lbox;
}

void SimpleSource::InitTracks(Track* track, int n) const {
	SimpleSourceKernel::InitTracksKernel(track, n, fEkin, fType, fLBox);
}