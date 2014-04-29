/*
 * Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <helper_math.h>
#include <helper_functions.h>
#include <helper_cuda.h>       // CUDA device initialization helper functions
#include "BilateralFilterCuda.hpp"

/*
    Perform a simple bilateral filter.

    Bilateral filter is a nonlinear filter that is a mixture of range
    filter and domain filter, the previous one preserves crisp edges and
    the latter one filters noise. The intensity value at each pixel in
    an image is replaced by a weighted average of intensity values from
    nearby pixels.

    The weight factor is calculated by the product of domain filter
    component(using the gaussian distribution as a spatial distance) as
    well as range filter component(Euclidean distance between center pixel
    and the current neighbor pixel). Because this process is nonlinear,
    the sample just uses a simple pixel by pixel step.

    Texture fetches automatically clamp to edge of image. 1D gaussian array
    is mapped to a 1D texture instead of using shared memory, which may
    cause severe bank conflict.

    Threads are y-pass(column-pass), because the output is coalesced.

    Parameters
    od - pointer to output data in global memory
    d_f - pointer to the 1D gaussian array
    e_d - euclidean delta
    w  - image width
    h  - image height
    r  - filter radius
*/

//// GLOBALS


__constant__ float cGaussian[64];   //gaussian array in device side

typedef texture<uchar4, 2, cudaReadModeNormalizedFloat> TextureU4f;
TextureU4f rgbaTex;

typedef texture<uchar4, 2, cudaReadModeNormalizedFloat> TextureU4f;
TextureU4f guideTex;

typedef texture<float, 2, cudaReadModeElementType> Texture32FC1;
Texture32FC1 depthTex_32FC1;

typedef texture<ushort, 2, cudaReadModeNormalizedFloat> Texture16UC1;
Texture16UC1 depthTex_16UC1;

// cost volume
//texture<float, 3, cudaReadModeNormalizedFloat> texCostVolume3D;  // 3D texture
//cudaArray *d_volumeArray = 0;

//// HELPERS

// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
}

__device__ float yangRangeDist( float4 a, float4 b, float d )
{

    float mod = ( fabs(b.x - a.x) +
                  fabs(b.y - a.y) +
                  fabs(b.z - a.z)  ) / 3.f;

    return __expf(-mod / d);
}

__device__ float euclideanLen( float a, float b, float d )
{

    float diff = (b - a);

    return __expf( -(diff*diff) / (2.f * d * d) );
}

__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(fabs(rgba.x));   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(fabs(rgba.y));
    rgba.z = __saturatef(fabs(rgba.z));
    rgba.w = __saturatef(fabs(rgba.w));
    return (uint(rgba.w * 255.0f) << 24) | (uint(rgba.z * 255.0f) << 16) | (uint(rgba.y * 255.0f) << 8) | uint(rgba.x * 255.0f);
}

__device__ float4 rgbaIntToFloat(uint c)
{
    float4 rgba;
    rgba.x = (c & 0xff) * 0.003921568627f;       //  /255.0f;
    rgba.y = ((c>>8) & 0xff) * 0.003921568627f;  //  /255.0f;
    rgba.z = ((c>>16) & 0xff) * 0.003921568627f; //  /255.0f;
    rgba.w = ((c>>24) & 0xff) * 0.003921568627f; //  /255.0f;
    return rgba;
}

//// PRECOMPUTATION

/*
    Because a 2D gaussian mask is symmetry in row and column,
    here only generate a 1D mask, and use the product by row
    and column index later.

    1D gaussian distribution :
        g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier

    parameters:
    og - output gaussian array in global memory
    delta - the 2nd parameter 'd' in the above function
    radius - half of the filter size
             (total filter size = 2 * radius + 1)
*/
extern "C"
void updateGaussian( float delta, int radius )
{
    float  fGaussian[64];

    for ( int i = 0; i < 2*radius + 1; ++i )
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta)); // orig
        //fGaussian[i] = expf(-(x*x) / (2*delta*delta));   // Yang?
    }

    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// CrossBilateral 32FC1
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// NOTE: there is no point using <ushort> textures,
// since that would normalise over 65536 and not 10001

/*
 * @brief Input type dependent texture read function
 * @param T     Texture selector (depthTex_32FC1, or depthTex_16UC1)
 * @param x     x coordinate to read from in texture
 * @param y     y coordinate to read from in texture
 */
template <typename T>
__device__
float fetchTexture( int x, int y );

template<> float fetchTexture<float>( int x, int y )
{
    return tex2D( depthTex_32FC1, x, y );
}

template<> float fetchTexture<ushort>( int x, int y )
{
    return tex2D( depthTex_16UC1, x, y );
}

/*
 * @brief Main crossfilter kernel
 * @param dOut              normalised float output memory
 * @param w                 texture width
 * @param h                 texture height
 * @param outPitch          elementcount of one row of dOut
 * @param costVolume        w x (h * costVolumeZDim) read/write global array
 * @param costVolumePitch   elementcount of one row of costVolume
 * @param costVolumeZDim    depth of costVolume
 * @param e_d               eucledian delta (range sigma)
 * @param r                 kernel half width
 */
template <typename T>
__global__ void
d_cross_bilateral_filterF( T *dOut, int w, int h, size_t outPitch,
                           //float *costVolume, size_t costVolumePitch, uint costVolumeZDim,
                           float e_d, int r, unsigned char fillMode = FILL_ALL | SKIP_ZEROS )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float t = 0.f;
    float4 guideCenter = tex2D( guideTex, x, y );
    T      centerPix   = fetchTexture<T>( x, y );

    // check for early exit
    if ( !(fillMode & FILL_ZEROS) && (centerPix != 0.f) ) // if 0, and NOT FILL_ZEROS
    {
        dOut[y * outPitch + x] = centerPix;
        return;
    }

    // estimate cost volume
    for ( int i = -r; i <= r; ++i )
    {
        for ( int j = -r; j <= r; ++j )
        {
            // read depth
            T curPix = fetchTexture<T>( x+j, y+i );

            // skip, if no data
            if ( (fillMode & SKIP_ZEROS) && (curPix == 0.f) )
                continue;

            // read rgb
            float4 guidePix = tex2D( guideTex, x + j, y + i );

            // estimate weight
            factor = cGaussian[i + r] * cGaussian[j + r] *       // spatial factor
                     //expf( -sqrt(i*i+j*j) / e_d ) *
                     yangRangeDist( guidePix, guideCenter, e_d ); // range   factor

            // accumulate
            t   += factor * curPix;
            sum += factor;
        }
    }
    if ( sum > 0.f )
        dOut[y * outPitch + x] = t / sum;
    else
        dOut[y * outPitch + x] = centerPix;
}

/*
 * @brief Texture binding based on input template type (float tested only)
 * @param texRefPtr Cuda reference pointer to one of the globals at top of the file.
 */
template <typename TImg>
void prepareInputTex( textureReference const*& );

// <float> expects dImage to be normalised float
template<>
void prepareInputTex<float>( textureReference const*& texRefPtr )
{
    cudaGetTextureReference( &texRefPtr, &depthTex_32FC1 );
}

// <ushort> expects dImage to be 0..65536
template<>
void prepareInputTex<ushort>( textureReference const*& texRefPtr )
{
    cudaGetTextureReference( &texRefPtr, &depthTex_16UC1 );
}

/*
 * @brief Cross biltareal filtering. Use <float> version, the others are untested.
 * @param dDest     Device pointer, currently giving normalised floats
 * @param dImage    Input pointer, currently expecting normalised floats
 * @param dTemp     Copy buffer for dImage
 * @param pitch     dImage and dTemp pitch, not used (since texturing)
 * @param dGuide    uchar4 XRGB image (0..255) read as normalised float through "guideTex"
 * @param guidePitch dGuide pitch, not used, since texturing
 * @param width     Width of every input, and output
 * @param height    Height of every input and output
 * @param e_d       Eucledian delta (range sigma)
 * @param radius    Kernel half width
 * @param iterations Not tested to be other, than one
 * @param timer     Performance timing
 */
template <typename T>
double crossBilateralFilterF( T *dDest, uint destPitch,
                              T *dImage, T *dTemp, uint imagePitch,
                              uint *dGuide, uint guidePitch,
                              //float *dCostVolume, uint costVolumePitch,
                              cudaExtent volumeSize,
                              float e_d, int radius, int iterations
                              , unsigned char fillMode,
                              StopWatchInterface *timer
                              )
{
    // var for kernel computation timing
    double dKernelTime;

    depthTex_16UC1.addressMode[0] = cudaAddressModeMirror;
    depthTex_16UC1.addressMode[1] = cudaAddressModeMirror;
    depthTex_32FC1.addressMode[0] = cudaAddressModeMirror;
    depthTex_32FC1.addressMode[1] = cudaAddressModeMirror;

    // bind input image texture
    textureReference const* texRefPtr;
    prepareInputTex<T>( texRefPtr );

    // Bind inpput image to the texture
    cudaChannelFormatDesc descT = cudaCreateChannelDesc<T>();
    size_t offset = 0;
    checkCudaErrors( cudaBindTexture2D(&offset, texRefPtr, dImage, &descT, volumeSize.width, volumeSize.height, imagePitch) );
    if ( offset > 0 ) std::cerr << "cudaBindTexture2D returned non-zero offset!!!" << std::endl;

    // Bind guide texture
    cudaChannelFormatDesc descU4 = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors( cudaBindTexture2D(&offset, guideTex, dGuide, descU4, volumeSize.width, volumeSize.height, guidePitch) );
    if ( offset > 0 )
    {
        std::cerr << "cudaBindTexture2D returne non-zero offset!!!" << std::endl;
    }

    // work
    for ( int i = 0; i < iterations; ++i )
    {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((volumeSize.width + 16 - 1) / 16, (volumeSize.height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_cross_bilateral_filterF<<< gridSize, blockSize>>>( dDest, volumeSize.width, volumeSize.height, destPitch / sizeof(T),
                                                             //dCostVolume, costVolumePitch / sizeof(float), volumeSize.depth,
                                                             e_d, radius,
                                                             (   (fillMode == FILL_ALL_THEN_FILL_ZEROS)
                                                               ? ( (i>0) ? (FILL_ZEROS | SKIP_ZEROS) : (FILL_ALL | SKIP_ZEROS) )
                                                               : fillMode )
                                                             ///* fillOnlyZeros: */ (fillMode == FILL_ALL) ? false :
                                                             //                            ( fillMode == FILL_ONLY_ZEROS ? true : (i>0) )
                                                                                                                );


        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, imagePitch, dDest, sizeof(T) * volumeSize.width,
                                         sizeof(T) * volumeSize.width, volumeSize.height, cudaMemcpyDeviceToDevice));

            //checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, cudaCreateChannelDesc<T>(), width, height, pitch));
            checkCudaErrors( cudaBindTexture2D(&offset, texRefPtr, dTemp, &descT, volumeSize.width, volumeSize.height, imagePitch) );
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}

/*
 * @brief Template specialisation declaration (needed by "extern" in BilateralFilteringCuda.hpp)
 */
template double crossBilateralFilterF( float *dDest,                uint destPitch,
                                       float *dImage, float *dTemp, uint imagePitch,
                                       uint  *dGuide,               uint guidePitch,
                                       //float *dCostVolume,          uint costVolumePitch,
                                       cudaExtent volumeSize,
                                       float e_d, int radius, int iterations, unsigned char fillOnlyZeros,
                                       StopWatchInterface *timer );

#if 0 // Not implemented yet...
template double crossBilateralFilterF( ushort *dDest,
                                       ushort *dImage, ushort *dTemp, uint pitch,
                                       unsigned *dGuide, unsigned guidePitch,
                                       int width, int height,
                                       float e_d, int radius, int iterations,
                                       StopWatchInterface *timer );
#endif

template <typename T>
__global__ void
d_cross_bilateral_filterCV( T *dOut, int w, int h, size_t outPitch,
                           float *costVolume, size_t costVolumePitch, uint costVolumeZDim,
                           float e_d, int r, bool onlyZeros = false )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float t = 0.f;
    float4 guideCenter = tex2D( guideTex, x, y );
    T      centerPix   = fetchTexture<T>( x, y );

    // check for early exit
    /*if ( onlyZeros && (centerPix != 0.f) )
    {
        dOut[y * outPitch + x] = centerPix;
        return;
    }*/

    // estimate cost volume
    for ( int z = 0; z < costVolumeZDim; ++z )
    {
        for ( int i = -r; i <= r; ++i )
        {
            for ( int j = -r; j <= r; ++j )
            {
                // read depth
                T curPix = fetchTexture<T>( x+j, y+i );
                // skip, if no data
                if ( curPix == 0.f )
                    continue;

                // read rgb
                float4 guidePix = tex2D( guideTex, x + j, y + i );

                // estimate weight
                factor = cGaussian[i + r] * cGaussian[j + r] *       // spatial factor
                         euclideanLen( guidePix, guideCenter, e_d ); // range   factor

                // accumulate
                t   += factor * curPix;
                sum += factor;
            }
        }

        // images are continuosly stored below each other in costVolume
        costVolume[ (z * h + y) * costVolumePitch + x ] = t / sum;

        // old
        if ( z == costVolumeZDim / 2 )
            dOut[y * outPitch + x] = t / sum;
    } // end for z
}


///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

////
///  Bilateral RGBA (8UC4)
////

__global__ void
d_bilateral_filterRGBA( uint *od, int w, int h, float e_d, int r )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float4 t = {0.f, 0.f, 0.f, 0.f};
    float4 center = tex2D(rgbaTex, x, y);

    for (int i = -r; i <= r; ++i)
    {
        for (int j = -r; j <= r; ++j)
        {
            float4 curPix = tex2D(rgbaTex, x + j, y + i);
            if ( curPix.x == 0 )
                continue;
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen(curPix, center, e_d);             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }

    od[y * w + x] = rgbaFloatToInt(t/sum);
}

/*
    Perform 2D bilateral filter on image using CUDA

    Parameters:
    d_dest - pointer to destination image in device memory
    width  - image width
    height - image height
    e_d    - euclidean delta
    radius - filter radius
    iterations - number of iterations
*/
extern "C"
double bilateralFilterRGBA(uint *dDest,
                           int width, int height,
                           float e_d, int radius, int iterations,
                           StopWatchInterface *timer,
                           uint* dImage, uint* dTemp, uint pitch )
{
    // var for kernel computation timing
    double dKernelTime;

    // Bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    size_t offset = 0;
    checkCudaErrors( cudaBindTexture2D(&offset, rgbaTex, dImage, desc, width, height, pitch) );
    if ( offset > 0 )
    {
        std::cerr << "cudaBindTexture2D returne non-zero offset!!!" << std::endl;
    }

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filterRGBA<<< gridSize, blockSize>>>(
                                                           dDest, width, height, e_d, radius );

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}

////
/// Bilateral 32FC1
////
__global__ void
d_bilateral_filterF( float *od, int w, int h, float e_d, int r )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float t = 0.f;
    float center = tex2D( depthTex_32FC1, x, y );

    for ( int i = -r; i <= r; ++i )
    {
        for ( int j = -r; j <= r; ++j )
        {
            float curPix = tex2D(depthTex_32FC1, x + j, y + i);

            if ( curPix == 0.f ) // skip, if empty
                continue;

            factor = cGaussian[i + r] * cGaussian[j + r] *     // domain factor
                     euclideanLen(curPix, center, e_d);        // range factor

            t   += factor * curPix;
            sum += factor;
        }
    }

    // output
    od[y * w + x] = t / sum;
}

extern "C"
double bilateralFilterF( float *dDest,
                         int width, int height,
                         float e_d, int radius, int iterations,
                         StopWatchInterface *timer,
                         float* dImage, float* dTemp, uint pitch )
{
    // var for kernel computation timing
    double dKernelTime;

    // Bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    size_t offset = 0;
    checkCudaErrors( cudaBindTexture2D(&offset, depthTex_32FC1, dImage, desc, width, height, pitch) );
    if ( offset > 0 )
    {
        std::cerr << "cudaBindTexture2D returne non-zero offset!!!" << std::endl;
    }

    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_bilateral_filterF<<< gridSize, blockSize>>>( dDest, width, height, e_d, radius );

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}

////
/// CrossBilateral RGBA (8UC4)
////
__global__ void
d_cross_bilateral_filterRGBA( uint *od, int w, int h, float e_d, int r )
{
    int x = blockIdx.x*blockDim.x + threadIdx.x;
    int y = blockIdx.y*blockDim.y + threadIdx.y;

    if (x >= w || y >= h)
    {
        return;
    }

    float sum = 0.0f;
    float factor;
    float4 t = {0.f, 0.f, 0.f, 0.f};
    float4 center = tex2D(guideTex, x, y);

    for (int i = -r; i <= r; ++i)
    {
        for (int j = -r; j <= r; ++j)
        {
            float4 curPix = tex2D( rgbaTex, x + j, y + i );
            float4 guidePix = tex2D( guideTex, x + j, y + i );
            if ( curPix.x == 0 )
                continue;
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen(guidePix, center, e_d);             //range factor

            t += factor * curPix;
            sum += factor;
        }
    }

    od[y * w + x] = rgbaFloatToInt(t/sum);
}

extern "C"
double crossBilateralFilterRGBA( uint *dDest,
                                 uint *dImage, uint *dTemp, uint pitch,
                                 uint *dGuide, uint guidePitch,
                                 int width, int height,
                                 float e_d, int radius, int iterations,
                                 StopWatchInterface *timer
                                 )
{
    // var for kernel computation timing
    double dKernelTime;

    // Bind the array to the texture
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<uchar4>();
    size_t offset = 0;
    checkCudaErrors( cudaBindTexture2D(&offset, rgbaTex, dImage, desc, width, height, pitch) );
    if ( offset > 0 )
    {
        std::cerr << "cudaBindTexture2D returne non-zero offset!!!" << std::endl;
    }

    checkCudaErrors( cudaBindTexture2D(&offset, guideTex, dGuide, desc, width, height, guidePitch) );
    if ( offset > 0 )
    {
        std::cerr << "cudaBindTexture2D returne non-zero offset!!!" << std::endl;
    }


    for (int i=0; i<iterations; i++)
    {
        // sync host and start kernel computation timer
        dKernelTime = 0.0;
        checkCudaErrors(cudaDeviceSynchronize());
        sdkResetTimer(&timer);

        dim3 gridSize((width + 16 - 1) / 16, (height + 16 - 1) / 16);
        dim3 blockSize(16, 16);
        d_cross_bilateral_filterRGBA<<< gridSize, blockSize>>>(
                                                                 dDest, width, height, e_d, radius );

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, desc, width, height, pitch));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}

/*
     // Cost volume
    {
        // create 3D array
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
        cudaExtent volumeSize = make_cudaExtent( width, height, 9 );
        checkCudaErrors( cudaMalloc3DArray(&d_volumeArray, &channelDesc, volumeSize) );

        // set texture parameters
        texCostVolume3D.normalized = false;                      // access with normalized texture coordinates
        texCostVolume3D.filterMode = cudaFilterModePoint;      // linear interpolation
        texCostVolume3D.addressMode[0] = cudaAddressModeBorder;   // wrap texture coordinates
        texCostVolume3D.addressMode[1] = cudaAddressModeBorder;
        texCostVolume3D.addressMode[2] = cudaAddressModeBorder;

        // bind array to 3D texture
        checkCudaErrors( cudaBindTextureToArray(texCostVolume3D, d_volumeArray, channelDesc) );
    }
*/
