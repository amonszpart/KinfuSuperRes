#include "cuda.h"
#include "helper_cuda.h"

// ************************************************************************* //
// SET KERNEL2D                                                            * //
// ************************************************************************* //

template<typename T>
__global__ void setKernel2D( T *mem, T value, int w, int h )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    mem[ y * w + x ] = value;
}

template<typename T>
void runSetKernel2D( T* memory, T value, unsigned width, unsigned height )
{
    checkCudaErrors( cudaDeviceSynchronize() );

    dim3 gridSize( (width  + 16 - 1) / 16,
                   (height + 16 - 1) / 16 );
    dim3 blockSize( 16, 16 );
    setKernel2D<<< gridSize, blockSize>>>( memory, value, width, height );

    // sync host and stop computation timer
    checkCudaErrors( cudaDeviceSynchronize() );
}

template void runSetKernel2D( float* memory, float value, unsigned width, unsigned height );

// ************************************************************************* //
// *  COPY KERNEL2D                                                        * //
// ************************************************************************* //

template <typename T>
__global__ void copyKernel2D( T *in,
                            T *out,
                            int w, int h,
                            size_t in_pitch,
                            size_t out_pitch )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    float px = in[ y * w + x ];

    out[ y * w + x ] = px;
}

template <typename T>
void runCopyKernel2D( T *in , unsigned w_in , unsigned h_in , size_t pitch_in,
                      T *out, size_t pitch_out )
{
    dim3 gridSize((w_in + 16 - 1) / 16, (h_in + 16 - 1) / 16);
    dim3 blockSize( 16, 16 );
    copyKernel2D<<< gridSize, blockSize>>>( in, out, w_in, h_in, pitch_in, pitch_out );

    // sync host and stop computation timer
    checkCudaErrors( cudaDeviceSynchronize() );
}

template void runCopyKernel2D<float>( float *in , unsigned w_in , unsigned h_in , size_t pitch_in,
                                      float *out, size_t pitch_out );
