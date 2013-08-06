// http://www.cl.cam.ac.uk/research/rainbow/projects/dcbgrid/DCBGrid-preprint.pdf

#include "GpuDepthMap.h"
#include "AmCudaHelper.cuh"
#include "helper_cuda.h"
//#include "CudaHelperCommon.cuh"

void GpuDepthMap::Create(GpuDepthMapType type, int width, int height)
{
    if ( type != mapType || width != w || height != h )
    {
        Destroy();

        //LOG_EVENT("Creating depth map");
        CUDA_CALL(cudaMallocPitch((void**)&gpuImage, &gpuImagePitch, width * sizeof(float), height));
        w = width;
        h = height;
        mapType = type;
    }
}

void GpuDepthMap::Destroy()
{
    //LOG_EVENT("Destroying depth map");
    CUDA_FREE(gpuImage);
    w = h = 0;
}

void GpuDepthMap::CopyDataOut( float* hostData )
{
    CUDA_CALL(cudaMemcpy2D( hostData, w * sizeof(float),
                            gpuImage, gpuImagePitch,
                            w * sizeof(float), h,
                            cudaMemcpyDeviceToHost ));
}

void GpuDepthMap::CopyDataIn( float* const hostData )
{
    checkCudaErrors(cudaMemcpy2D( gpuImage, gpuImagePitch, hostData,
                                  w * sizeof(float), w * sizeof(float), h,
                                  cudaMemcpyHostToDevice ));
}

void GpuDepthMap::SwapData(GpuDepthMap & other)
{
    std::swap<float*>(gpuImage, other.gpuImage);
    std::swap<size_t>(gpuImagePitch, other.gpuImagePitch);
    std::swap<GpuDepthMapType>(mapType, other.mapType);
    std::swap<int>(w, other.w);
    std::swap<int>(h, other.h);
}
