// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <helper_cuda.h>
//#include <helper_math.h>
//#include <helper_functions.h>

#include "GpuDepthMap.hpp"

////
/// SQUAREDIFF
////

texture<float, 2, cudaReadModeElementType> tex_fDep;

__global__ void squareDiffKernel( float *d_C2,
                                  const int w, const int h, const size_t C2_pitch,
                                  const float d, const float truncAt )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (x >= w) || (y >= h) ) return;

    // read
    const float d_in = tex2D( tex_fDep, x, y );
    // out
    d_C2[ y * C2_pitch + x ] = min( (d_in - d) * (d_in - d), truncAt );
}

void squareDiff( GpuDepthMap<float> const& d_fDep, float d,
                 GpuDepthMap<float>      & d_C2  , float truncAt )
{
    // Bind inpput image to the texture
    cudaChannelFormatDesc descT = cudaCreateChannelDesc<float>();
    size_t offset = 0;
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_fDep, d_fDep.Get(), &descT,
                                       d_fDep.GetWidth(), d_fDep.GetHeight(), d_fDep.GetPitch() ));
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // launch
    dim3 gridSize((d_fDep.GetWidth() + 16 - 1) / 16, (d_fDep.GetHeight() + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    squareDiffKernel<<<gridSize, blockSize>>>( d_C2.Get(),
                                               d_C2.GetWidth(), d_C2.GetHeight(), d_C2.GetPitch()/sizeof(float),
                                               d, truncAt );

    // sync host and stop computation timer
    checkCudaErrors(cudaDeviceSynchronize());
}

////
/// MinMaskedCopy
////
texture<float, 2, cudaReadModeElementType> tex_C;
texture<float, 2, cudaReadModeElementType> tex_Cprev;

__global__ void minMaskedCopyKernel(
        float *minC  , const size_t minC_pitch  ,
        float *minCm1, const size_t minCm1_pitch,
        float *minCp1, const size_t minCp1_pitch,
        float *minDs , const size_t minDs_pitch ,
        const int w, const int h,
        const float d )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (x >= w) || (y >= h) ) return;

    // read
    const float c     = tex2D( tex_C, x, y );
    const float c_min = minC [ y * minC_pitch + x ];
    //const float d_min = minDs[ y * minDs_pitch + x ];

    if ( c < c_min )                                    // if ( C[i] < minC[i] )
    {
        const float c_prev = tex2D( tex_Cprev, x, y );

        // store current C(d)
        minC  [ y * minC_pitch   + x ] = c;        // minC[i] = C[i]
        // store current d
        minDs [ y * minDs_pitch  + x ] = d;        // minDs[i] = d
        // copy previous d's cost to (d-1)'s cost
        minCm1[ y * minCm1_pitch + x ] = c_prev;   // minCm1[i] = Cprev[i]
    }
    else if ( minDs[y * minDs_pitch + x] == d-1 )                       // if ( minDs[i] == d-1 )
    {
        minCp1[ y * minCp1_pitch + x ] = c;        // minCm1[i] = Cprev[i]
    }
}

void minMaskedCopy( GpuDepthMap<float> const& C,
                    GpuDepthMap<float> const& Cprev,
                    GpuDepthMap<float>      & minC,
                    GpuDepthMap<float>      & minCm1,
                    GpuDepthMap<float>      & minCp1,
                    GpuDepthMap<float>      & minDs,
                    const float d )
{
    /// Bind inpput image to the texture
    // tex_C <- C
    cudaChannelFormatDesc descT = cudaCreateChannelDesc<float>();
    size_t offset = 0;
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_C, C.Get(), &descT,
                                       C.GetWidth(), C.GetHeight(), C.GetPitch() ));
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // tex_Cprev <- Cprev
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_Cprev, Cprev.Get(), &descT,
                                       Cprev.GetWidth(), Cprev.GetHeight(), Cprev.GetPitch() ));
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // launch
    dim3 gridSize((minC.GetWidth() + 16 - 1) / 16, (minC.GetHeight() + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    minMaskedCopyKernel<<<gridSize, blockSize>>>( minC  .Get(), minC  .GetPitch()/sizeof(float),
                                                  minCm1.Get(), minCm1.GetPitch()/sizeof(float),
                                                  minCp1.Get(), minCp1.GetPitch()/sizeof(float),
                                                  minDs .Get(), minDs .GetPitch()/sizeof(float),
                                                  minC.GetWidth(), minC.GetHeight(),
                                                  d );

    // sync host and stop computation timer
    checkCudaErrors(cudaDeviceSynchronize());
}

////
/// SubpixelRefine
////

texture<float, 2, cudaReadModeElementType> tex_minC;
texture<float, 2, cudaReadModeElementType> tex_minCm1;
texture<float, 2, cudaReadModeElementType> tex_minCp1;
texture<float, 2, cudaReadModeElementType> tex_minDs;

__global__ void subpixelRefineKernel(
        float *fDep_next, const size_t fDep_next_pitch,
        const int w, const int h )
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (x >= w) || (y >= h) ) return;

    // read
    const float c_min    = tex2D( tex_minC  , x, y );
    const float c_min_m1 = tex2D( tex_minCm1, x, y );
    const float c_min_p1 = tex2D( tex_minCp1, x, y );
    const float d_min    = tex2D( tex_minDs , x, y );

    // calculate
    float a1 = c_min_p1 - c_min_m1;
    float a2 = ( 2.f * (c_min_p1 + c_min_m1 - 2.f * c_min) );
    float a3 = a1 / a2;
    if ( d_min > 0 )
        a3 = min( max( a3, d_min * -3.f), d_min * 3.f );
    else
        a3 = min( max( a3, -10.f), 10.f );

    // out
    fDep_next[ y * fDep_next_pitch + x ] = d_min - a3;
}

void subpixelRefine( GpuDepthMap<float> const& minC  ,
                     GpuDepthMap<float> const& minCm1,
                     GpuDepthMap<float> const& minCp1,
                     GpuDepthMap<float> const& minDs,
                     GpuDepthMap<float>      & fDep_next )
{
    /// Bind inpput image to the texture
    // tex_minC <- minC
    cudaChannelFormatDesc descT  = cudaCreateChannelDesc<float>();
    size_t                offset = 0;
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_minC, minC.Get(), &descT,
                                       minC.GetWidth(), minC.GetHeight(), minC.GetPitch() ));
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // tex_minCm1 <- minCm1
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_minCm1, minCm1.Get(), &descT,
                                       minCm1.GetWidth(), minCm1.GetHeight(), minCm1.GetPitch() ));
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // tex_minCp1 <- minCp1
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_minCp1, minCp1.Get(), &descT,
                                       minCp1.GetWidth(), minCp1.GetHeight(), minCp1.GetPitch() ));
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // tex_minDs <- minDs
    checkCudaErrors(cudaBindTexture2D( &offset, &tex_minDs, minDs.Get(), &descT,
                                       minDs.GetWidth(), minDs.GetHeight(), minDs.GetPitch() ));

    // launch
    dim3 gridSize((fDep_next.GetWidth() + 16 - 1) / 16, (fDep_next.GetHeight() + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    subpixelRefineKernel<<<gridSize, blockSize>>>( fDep_next.Get(), fDep_next.GetPitch()/sizeof(float),
                                                   fDep_next.GetWidth(), fDep_next.GetHeight() );

    // sync host and stop computation timer
    checkCudaErrors(cudaDeviceSynchronize());
}

#if 0
////
/// Max
////

template <typename T>
__global__ void getMaxKernel( T const* in,
                              int pitch,
                              int w, int h,
                              T* out
                              )
{

    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    if ( (x >= w) || (y >= h) ) return;

    atomicMax( &out, in[y * pitch + x] );
}


template <typename T>
T getMax( GpuDepthMap<T> const& img )
{
    // launch
    dim3 gridSize((img.GetWidth() + 16 - 1) / 16, (img.GetHeight() + 16 - 1) / 16);
    dim3 blockSize(16, 16);
    GpuDepthMap<T> theMax;
    theMax.Create( DEPTH_MAP_TYPE_FLOAT, 1, 1 );
    T tmp[1];
    tmp[0] = 0;
    theMax.CopyDataIn( tmp );
    getMaxKernel<<<gridSize, blockSize>>>( img.Get(), img.GetPitch() / sizeof(T), img.GetWidth(), img.GetHeight(),
                                           theMax.Get() );
    theMax.CopyDataOut( tmp );
    return tmp[0];
}

template float getMax<float>( GpuDepthMap<float> const& img );
#endif
