#include "TrackStack.hh"
#include <cassert>
#include "Source.h"
#include "Utils.h"

TrackStack h_PhotonStack;
TrackStack h_ElectronStack;
__device__ TrackStack d_PhotonStack;
__device__ TrackStack d_ElectronStack;

TrackSeq h_TrackSeq;
__device__ TrackSeq d_TrackSeq;

void TrackStack::init(int maxsz) {
	fTop = -1;
	fCapacity = maxsz;
	cudaMalloc(&fData, fCapacity * sizeof(Track));
	CudaCheckError();
}

void TrackStack::release() {
	cudaFree(fData);
}

__host__ void TrackStack::push(const Track& hist) {
	//assert(!full());
	++fTop;
	cudaMemcpy(fData + fTop, &hist, sizeof(Track), cudaMemcpyHostToDevice);
}

__device__ Track& TrackStack::push_one() {
	assert(!full());
	int t = atomicAdd(&fTop, 1);
	return fData[t + 1];
}

void TrackStack::pop_n(Track* target, int n) {
	cudaMemcpy(target, fData + (fTop - n + 1),
		n * sizeof(Track), cudaMemcpyDeviceToDevice);
	fTop -= n;
}

void TrackSeq::init(int maxsz) {
	fSize = 0;
	fCapacity = maxsz;
	cudaMalloc(&fData, fCapacity * sizeof(Track));
	CudaCheckError();
}

void TrackSeq::release() {
	cudaFree(fData);
}

void TrackSeq::add_n_primary(int n, Source* source) {
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