#include "TrackStack.hh"
#include <cassert>
#include "Source.hh"

TrackStack h_PhotonStack;
TrackStack h_ElectronStack;
TrackSeq h_TrackSeq;

void TrackStack::init(int maxsz) {
	fTop = -1;
	fCapacity = maxsz;
	fData = (Track*)malloc(fCapacity * sizeof(Track));
}

void TrackStack::release() {
	free(fData);
}


Track& TrackStack::push_one() {
	assert(!full());
	int t = fTop;
	fTop++;
	return fData[t + 1];
}

void TrackStack::pop_n(Track* target, int n) {
	memcpy(target, fData + (fTop - n + 1), n * sizeof(Track));
	fTop -= n;
}

void TrackSeq::init(int maxsz) {
	fSize = 0;
	fCapacity = maxsz;
	fData = (Track*)malloc(fCapacity * sizeof(Track));
}

void TrackSeq::release() {
	free(fData);
}

void TrackSeq::add_n_primary(int n, const Source* source) {
	source->InitTracks(h_TrackSeq.fData + h_TrackSeq.fSize, n);
	fSize += n;
}

void TrackSeq::add_secondary(TrackStack* stack) {
	if (stack->size() > fCapacity) {
		stack->pop_n(h_TrackSeq.fData + fSize, fCapacity);
		fSize += fCapacity;
	}
	else {
		int stackSz = stack->size();	// pop_n will modify the stack->fSize
		stack->pop_n(h_TrackSeq.fData + fSize, stack->size());
		fSize += stackSz;
	}
}