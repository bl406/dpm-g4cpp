#pragma once

class Track;


class Source {
public:
	Source() {}
	virtual ~Source() {}

	virtual void InitTracks(Track* track, int n) const = 0;
};

class SimpleSource : public Source
{	
public:
	SimpleSource(float lbox);
	~SimpleSource() {}

	void InitTracks(Track* track, int n) const override;

private:
	float fEkin;
	int fType;
	float fLBox;
};

