#ifndef ERROR_CHECKING_H
#define ERROR_CHECKING_H

// cf. http://choorucode.wordpress.com/2011/03/02/cuda-error-checking/

#include <iostream>
#include <cstdlib>
#include <cublas_v2.h>
#include <cusparse.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <device_launch_parameters.h>

#define CudaSafeCall( err )     __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()        __cudaCheckError( __FILE__, __LINE__ )
#define CublasSafeCall( err )   __cublasSafeCall( err, __FILE__, __LINE__ )
#define CusparseSafeCall( err ) __cusparseSafeCall( err, __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError err, const char *file, const int line)
{
	if (cudaSuccess != err)
	{
		std::cerr << "CudaSafeCall() failed at " << file << ":" << line << ": " << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		std::exit(EXIT_FAILURE);
	}
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err)
	{
		std::cerr << "CudaCheckError() failed at " << file << ":" << line << ": " << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		system("pause");
		std::exit(EXIT_FAILURE);
	}

	err = cudaDeviceSynchronize();
	if (cudaSuccess != err)
	{
		std::cerr << "CudaCheckError() with sync failed at " << file << ":" << line << ": " << std::endl;
		std::cerr << cudaGetErrorString(err) << std::endl;
		system("pause");
		std::exit(EXIT_FAILURE);
	}

	return;
}

inline void __cublasSafeCall(cublasStatus_t status, const char* file, const int line)
{
	if(status != CUBLAS_STATUS_SUCCESS){
		std::cerr << "CublasSafeCall() failed at " << file << ":" << line << std::endl;
		switch (status)
		{
		case CUBLAS_STATUS_ALLOC_FAILED:
			std::cerr << "Alloc failed" << std::endl;
			break;
		case CUBLAS_STATUS_ARCH_MISMATCH:
			std::cerr << "Arch mismatch" << std::endl;
			break;
		case CUBLAS_STATUS_EXECUTION_FAILED:
			std::cerr << "Execution failed" << std::endl;
			break;
		case CUBLAS_STATUS_INTERNAL_ERROR:
			std::cerr << "Internal error" << std::endl;
			break;
		case CUBLAS_STATUS_INVALID_VALUE:
			std::cerr << "Invalid value" << std::endl;
			break;
		case CUBLAS_STATUS_MAPPING_ERROR:
			std::cerr << "Mapping error" << std::endl;
			break;
		case CUBLAS_STATUS_NOT_INITIALIZED:
			std::cerr << "Not initialized" << std::endl;
			break;
		}

		std::exit(EXIT_FAILURE);
	}
	return;
}

inline void __cusparseSafeCall(cusparseStatus_t status, const char *file, const int line)
{
	if (status != CUSPARSE_STATUS_SUCCESS)
	{
		std::cerr << "CusparseSafeCall() failed at " << file << ":" << line << std::endl;
		switch (status)
		{
		case CUSPARSE_STATUS_ALLOC_FAILED:
			std::cerr << "Alloc failed" << std::endl;
			break;
		case CUSPARSE_STATUS_ARCH_MISMATCH:
			std::cerr << "Arch mismatch" << std::endl;
			break;
		case CUSPARSE_STATUS_EXECUTION_FAILED:
			std::cerr << "Execution failed" << std::endl;
			break;
		case CUSPARSE_STATUS_INTERNAL_ERROR:
			std::cerr << "Internal error" << std::endl;
			break;
		case CUSPARSE_STATUS_INVALID_VALUE:
			std::cerr << "Invalid value" << std::endl;
			break;
		case CUSPARSE_STATUS_MAPPING_ERROR:
			std::cerr << "Mapping error" << std::endl;
			break;
		case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
			std::cerr << "Matrix type not supported" << std::endl;
			break;
		case CUSPARSE_STATUS_NOT_INITIALIZED:
			std::cerr << "Not initialized" << std::endl;
			break;
		}

		exit(EXIT_FAILURE);
	}
	return;
}

#endif /* end of include guard: ERROR_CHECKING_H */
