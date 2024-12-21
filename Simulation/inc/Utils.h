#pragma once

#include <vector>
#include <array>
#include <cuda_runtime.h>
#include "error_checking.h"
#include "config.hh"

template <class T>
std::array<double, 2> CalMeanAndStd(const std::vector<T>& data) {
	double sum = 0.0;
	for (auto& d : data) {
		sum += d;
	}
	double mean = sum / data.size();
	double sq_sum = 0.0;
	for (auto& d : data) {
		sq_sum += (d - mean) * (d - mean);
	}
	double std = std::sqrt(sq_sum / data.size());
	return { mean, std };
}

template <typename T>
void initCudaTexture(T* hostData, int* dims, int ndim, cudaTextureDesc* texDesc,
    cudaTextureObject_t& texObj, cudaArray_t& cuArray) {
    // ����CUDA����������
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<T>();

    if (ndim == 1) {
        // 1D����
        cudaMallocArray(&cuArray, &channelDesc, dims[0]);

        cudaMemcpyToArray(cuArray, 0, 0, hostData, dims[0] * sizeof(T), cudaMemcpyHostToDevice);
    }
    else if (ndim == 2) {
        // ����2D CUDA����
        cudaMallocArray(&cuArray, &channelDesc, dims[0], dims[1]);

        // ���������ݸ��Ƶ�2D CUDA����
		cudaMemcpy2DToArray(cuArray, 0, 0, hostData, dims[0] * sizeof(T), dims[0] * sizeof(T), dims[1], cudaMemcpyHostToDevice);      
    }
    else {
        // ����3D CUDA����
        cudaExtent extent = make_cudaExtent(dims[0], dims[1], dims[2]);
        cudaMalloc3DArray(&cuArray, &channelDesc, extent);

        // ���������ݸ��Ƶ�3D CUDA����
        cudaMemcpy3DParms copyParams = { 0 };
        copyParams.srcPtr = make_cudaPitchedPtr(hostData, dims[0] * sizeof(T), dims[0], dims[1]);
        copyParams.dstArray = cuArray;
        copyParams.extent = extent;
        copyParams.kind = cudaMemcpyHostToDevice;
        cudaMemcpy3D(&copyParams);
    }

    CudaCheckError();

    // ������Դ������
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = cuArray;

    // �����������
    cudaCreateTextureObject(&texObj, &resDesc, texDesc, nullptr);

    CudaCheckError();
}

inline int divUp(int total, int grain) {
	return (total + grain - 1) / grain;
}