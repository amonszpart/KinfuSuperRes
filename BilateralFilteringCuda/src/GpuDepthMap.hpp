#ifndef __GPU_DEPTH_MAP_H
#define __GPU_DEPTH_MAP_H
#pragma once

#include "AmCudaHelper.cuh"
#include <stddef.h>

#include <cuda_runtime.h>
#include "driver_types.h"
#include "helper_cuda.h"

enum GpuDepthMapType
{
    DEPTH_MAP_TYPE_ANY,
    DEPTH_MAP_TYPE_FLOAT,
    DEPTH_MAP_TYPE_USHORT
};

// A class encapsulating a depth-map, very similar to RgbImage
template <typename T>
class GpuDepthMap
{
    public:
        GpuDepthMap()
            : gpuImage(0), gpuImagePitch(0), w(0), h(0) {}
        ~GpuDepthMap() { Destroy(); }

        void Create( GpuDepthMapType type, int width, int height );
        void Destroy();

        T*       Get      ()       { return gpuImage;                      }
        const T* Get      () const { return gpuImage;                      }
        GpuDepthMapType GetType() const { return mapType; }

        int GetPitch      () const { return gpuImagePitch;                 }
        int GetWidth      () const { return w;                             }
        int GetHeight     () const { return h;                             }
        int getSizeInBytes() const { return h * gpuImagePitch * sizeof(T); }

        // Copies the contents of the depth map to host memory (which should be of the correct size)
        void CopyDataOut( T *      hostData );
        void CopyDataIn ( T *const hostData );

        /* This method allows nodes to operate without requiring in-place modification, swapping data
         * and responsibilities with the 'other' depth map */
        void SwapData( GpuDepthMap & other );

    private:
        T* gpuImage;
        size_t gpuImagePitch;
        int w, h;
        //size_t m_elemSize;

        GpuDepthMapType mapType;
};

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

#endif // __GPU_DEPTH_MAP_H
