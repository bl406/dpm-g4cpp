
#ifndef TrackStack_HH
#define TrackStack_HH

//
// A simple (singletone) track-stack to handle both primary and secondary
// particles. Each primary is inserted first, then each history goes till the
// stack becomes empty (i.e. PopIntoThisTrack() returns with -1). During the
// history, seconday tracks can be insterted by using the Insert() method (that
// makes sure that stack is deep enough and calles the Track::Reset() method
// before giving back a reference to the Track.
//

#include "Track.hh"

#include <vector>
#include <iostream>

class Source;

class TrackStack {
public:
	TrackStack() = default;
	~TrackStack() = default;

	void init(int maxsz);
	void release();

	int size() {
		return fTop + 1;
	}

	bool empty() {
		return fTop == -1;
	}

	bool full() {
		return fTop == fCapacity - 1;
	}

	Track& push_one();

	void pop_n(Track* target, int n);

	int fCapacity;
	int fTop;
	Track* fData;
};

class TrackSeq {
public:
	TrackSeq() = default;
	~TrackSeq() = default;

	void init(int maxsz);

	void release();

	void add_n_primary(int n, const Source* source);
	void add_secondary(TrackStack* stack);

	int size() {
		return fSize;
	}

	bool empty() {
		return fSize == 0;
	}
	
	bool full() {
		return fSize == fCapacity;
	}

	int fCapacity;
	int fSize;
	Track* fData;
};

extern TrackStack h_PhotonStack;
extern TrackStack h_ElectronStack;
extern TrackSeq h_TrackSeq;

#endif // TrackStack_HH
