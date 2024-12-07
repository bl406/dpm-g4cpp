#pragma once

class Track;

namespace SimpleSourceKernel {
	__global__ void InitTracksKernel(Track* track, int n, int ekin, int type);
}

class Source {
public:
	Source() {}
	virtual ~Source() {}

	virtual void InitTracks(Track* track, int n) const = 0;
};

class SimpleSource : public Source
{	
public:
	SimpleSource(float ekin, int type, float lbox);
	~SimpleSource() {}

	void InitTracks(Track* track, int n) const override;

private:
	float fEkin;
	int fType;
	float fLBox;
};

