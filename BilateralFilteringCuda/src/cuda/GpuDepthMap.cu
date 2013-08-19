// http://www.cl.cam.ac.uk/research/rainbow/projects/dcbgrid/DCBGrid-preprint.pdf

#include "GpuDepthMap.hpp"
#include "AmCudaHelper.cuh"
#include "helper_cuda.h"
//#include "CudaHelperCommon.cuh"

#if 0
template<typename T>
void GpuDepthMap<T>::Create( GpuDepthMapType type, int width, int height )
{
    if ( type != mapType || width != w || height != h )
    {
        Destroy();

        //LOG_EVENT("Creating depth map");
        CUDA_CALL( cudaMallocPitch((void**)&gpuImage, &gpuImagePitch, width * sizeof(T), height) );
        w = width;
        h = height;
        mapType = type;
    }
}

template<typename T>
void GpuDepthMap<T>::Destroy()
{
    //LOG_EVENT("Destroying depth map");
    CUDA_FREE(gpuImage);
    w = h = 0;
}

template<typename T>
void GpuDepthMap<T>::CopyDataOut( T* hostData )
{
    CUDA_CALL(cudaMemcpy2D( hostData, w * sizeof(T),
                            gpuImage, gpuImagePitch,
                            w * sizeof(T), h,
                            cudaMemcpyDeviceToHost ));
}

template<typename T>
void GpuDepthMap<T>::CopyDataIn( T* const hostData )
{
    checkCudaErrors(cudaMemcpy2D( gpuImage, gpuImagePitch, hostData,
                                  w * sizeof(T), w * sizeof(T), h,
                                  cudaMemcpyHostToDevice ));
}

template<typename T>
void GpuDepthMap<T>::SwapData(GpuDepthMap & other)
{
    std::swap<T*>(gpuImage, other.gpuImage);
    std::swap<size_t>(gpuImagePitch, other.gpuImagePitch);
    std::swap<GpuDepthMapType>(mapType, other.mapType);
    std::swap<int>(w, other.w);
    std::swap<int>(h, other.h);
}
#endif
