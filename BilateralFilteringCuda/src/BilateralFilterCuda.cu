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

typedef texture<float, 2, cudaReadModeElementType> TextureU1f;
TextureU1f depthTex;

//// HELPERS

// Euclidean Distance (x, y, d) = exp((|x - y| / d)^2 / 2)
__device__ float euclideanLen(float4 a, float4 b, float d)
{

    float mod = (b.x - a.x) * (b.x - a.x) +
                (b.y - a.y) * (b.y - a.y) +
                (b.z - a.z) * (b.z - a.z);

    return __expf(-mod / (2.f * d * d));
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
void updateGaussian(float delta, int radius)
{
    float  fGaussian[64];

    for (int i = 0; i < 2*radius + 1; ++i)
    {
        float x = i-radius;
        fGaussian[i] = expf(-(x*x) / (2*delta*delta));
    }

    checkCudaErrors(cudaMemcpyToSymbol(cGaussian, fGaussian, sizeof(float)*(2*radius+1)));
}

//// Bilateral RGBA (8UC4)

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
    size_t offset = -1;
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

//// Bilateral 32FC1

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
    float center = tex2D( depthTex, x, y );

    for ( int i = -r; i <= r; ++i )
    {
        for ( int j = -r; j <= r; ++j )
        {
            float curPix = tex2D(depthTex, x + j, y + i);

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
    size_t offset = -1;
    checkCudaErrors( cudaBindTexture2D(&offset, depthTex, dImage, desc, width, height, pitch) );
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


//// CrossBilateral RGBA (8UC4)

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
    size_t offset = -1;
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


//// CrossBilateral 32FC1

__global__ void
d_cross_bilateral_filterF( float *od, int w, int h, float e_d, int r )
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
    float4 center = tex2D(guideTex, x, y);

    for ( int i = -r; i <= r; ++i )
    {
        for ( int j = -r; j <= r; ++j )
        {
            float curPix   = tex2D( depthTex, x + j, y + i );
            float4 guidePix = tex2D( guideTex, x + j, y + i );
            if ( curPix == 0.f )
                continue;
            factor = cGaussian[i + r] * cGaussian[j + r] *     //domain factor
                     euclideanLen( guidePix, center, e_d );             //range factor

            t   += factor * curPix;
            sum += factor;
        }
    }

    od[y * w + x] = t / sum;
    //od[y * w + x] = tex2D( depthTex, x , y );
}

extern "C"
double crossBilateralFilterF( float *dDest,
                              float *dImage, float *dTemp, uint pitch,
                              uint *dGuide, uint guidePitch,
                              int width, int height,
                              float e_d, int radius, int iterations,
                              StopWatchInterface *timer
                              )
{
    // var for kernel computation timing
    double dKernelTime;

    // Bind the array to the texture
    cudaChannelFormatDesc descF = cudaCreateChannelDesc<float>();
    size_t offset = -1;
    checkCudaErrors( cudaBindTexture2D(&offset, depthTex, dImage, descF, width, height, pitch) );
    if ( offset > 0 )
    {
        std::cerr << "cudaBindTexture2D returne non-zero offset!!!" << std::endl;
    }

    cudaChannelFormatDesc descU4 = cudaCreateChannelDesc<uchar4>();
    checkCudaErrors( cudaBindTexture2D(&offset, guideTex, dGuide, descU4, width, height, guidePitch) );
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
        d_cross_bilateral_filterF<<< gridSize, blockSize>>>( dDest, width, height, e_d, radius );

        // sync host and stop computation timer
        checkCudaErrors(cudaDeviceSynchronize());
        dKernelTime += sdkGetTimerValue(&timer);

        if (iterations > 1)
        {
            // copy result back from global memory to array
            checkCudaErrors(cudaMemcpy2D(dTemp, pitch, dDest, sizeof(int)*width,
                                         sizeof(int)*width, height, cudaMemcpyDeviceToDevice));
            checkCudaErrors(cudaBindTexture2D(0, rgbaTex, dTemp, descF, width, height, pitch));
        }
    }

    return ((dKernelTime/1000.)/(double)iterations);
}


