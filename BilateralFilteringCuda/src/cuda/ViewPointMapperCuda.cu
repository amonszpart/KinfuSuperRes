// Source: https://github.com/jtbates/lua---kinect/blob/master/depth_to_point_cloud_par.cu
//#include "/media/Storage/workspace_ubuntu/rec/imgs_20130801_1607_calibPrism3/calibration.h"
//#include "/media/Storage/workspace_ubuntu/rec/imgs_20130805_1047_calibPrism4/calibration.h"
#include "calibration_cuda_constants_prism4.h"

#include "GpuDepthMap.hpp"
#include "cutil_math.cuh"
#include "AMCudaUtil.cu"
#include "AmCudaHelper.cuh"

#include "cuda.h"
#include "helper_cuda.h"

#include <iostream>

// comp_distortion_oulu.m rewrite from Bogouet toolbox
/*
 * \return normalized coordinates { xn.x, xn.y, 1.f }
 */
__device__ float3 cam2World( float2 xd,
                             float2 F, float2 C,
                             float k1, // K1
                             float k2, // K2
                             float p1, // K3
                             float p2, // K4
                             float k3, // K5
                             float alpha = 0.f )
{
    xd.x = (xd.x - C.x) / F.x;
    xd.y = (xd.y - C.y) / F.y;

    // initial guess
    float2 xn = xd;

    // skew
    xn.x -= alpha * xn.y;

    for ( int kk = 20; kk; --kk )
    {
        float r2        = xn.x * xn.x +
                          xn.y * xn.y;
        float sqr_r_2   = r2 * r2;
        float k_radial  = 1.f +
                          k1 * r2           +
                          k2 * sqr_r_2      +
                          k3 * sqr_r_2 * r2;

        float2 delta_x = { 2.f * p1 * xn.x * xn.y + p2 * (r2 + 2.f * xn.x * xn.x),
                           2.f * p2 * xn.x * xn.y + p1 * (r2 + 2.f * xn.y * xn.y)  };

        xn = (xd - delta_x) / (float2) { k_radial, k_radial };
    }

    return (float3){ xn.x, xn.y, 1.f };
}

__device__ float2 world2Cam( float3 xw,
                             float2 F, float2 C,
                             float k1, // K1
                             float k2, // K2
                             float p1, // K3
                             float p2, // K4
                             float k3, // K5
                             float alpha = 0.f)
{
    float2 x = {
        xw.x / xw.z,
        xw.y / xw.z };

    float r2 = x.x*x.x + x.y*x.y;
    float r4 = r2*r2;
    float r6 = r4 * r2;

    // Radial distortion:
    float cdist = 1.f + k1 * r2 + k2 * r4 + k3 * r6;
    float2 xd1 = x * cdist;

    // tangential distortion:
    float a1 = 2.f * x.x * x.y;
    float a2 = r2 + 2.f * x.x * x.x;
    float a3 = r2 + 2 * x.y * x.y;

    float2 delta_x = { p1 * a1 + p2 * a2,
                       p1 * a3 + p2 * a1 };

    float2 xd2 = xd1 + delta_x;

    // skew
    float2 xd3 = { xd2.x + alpha * xd2.y,
                   xd2.y };

    // to camera space
    return xd3 * F + C;
}

// Args:
//   depth_abs - the absolute depth from the kinect.
//   depth_proj - the projected depth.
template <typename T>
__global__ void mapViewpointKernel( const T *in,
                                          T *out,
                                    int w, int h,
                                    size_t in_pitch,
                                    size_t out_pitch )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    const float d = in[ y * in_pitch + x ]; // * 10001.f

    float3 P_world_left = d * cam2World( (float2){x   , y   },
                                         (float2){FX_D, FY_D}, (float2){CX_D, CY_D},
                                         K1_D, K2_D, K3_D, K4_D, K5_D,
                                         ALPHA_D );
    // R * [X; Y; Z] + T
    float3 P_world_right = {
        (R1 * P_world_left.x) + (R2 * P_world_left.y) + (R3 * P_world_left.z) + T1,
        (R4 * P_world_left.x) + (R5 * P_world_left.y) + (R6 * P_world_left.z) + T2,
        (R7 * P_world_left.x) + (R8 * P_world_left.y) + (R9 * P_world_left.z) + T3 };

    float2 p2_right = world2Cam( P_world_right,
                                 (float2){FX_RGB, FY_RGB}, (float2){CX_RGB, CY_RGB},
                                 K1_RGB, K2_RGB, K3_RGB, K4_RGB, K5_RGB,
                                 ALPHA_RGB );

    // map to 0.f..1.f
    //P_world_right.z /= 10001.f;

    // create int pointer
    int *pZ = (int*) &(P_world_right.z);

    // round to integer coordinates
    int2 int_p_right = { nearbyintf(p2_right.x),
                         nearbyintf(p2_right.y) };

    // check boundaries
    if ( (int_p_right.x >= w) || (int_p_right.y >= h) ||
         (int_p_right.x <  0) || (int_p_right.y <  0)   ) return;

    float old = atomicCAS ( (int*)&(out[ int_p_right.y * out_pitch + int_p_right.x ]), 0, *pZ );
    if ( old != 0.f )
    {
        atomicMin( (int*)&(out[ int_p_right.y * out_pitch + int_p_right.x ]), *pZ );
    }
}

//// runMapViewpoint ////

template<typename T>
void runMapViewpoint( GpuDepthMap<T> const& in, GpuDepthMap<T> &out )
{
    cudaMemset( out.Get(), 0, out.GetWidth() * out.GetHeight() * sizeof(T) );
    checkCudaErrors( cudaDeviceSynchronize() );

    dim3 gridSize( (in.GetWidth()  + 16 - 1) / 16,
                   (in.GetHeight() + 16 - 1) / 16 );
    dim3 blockSize( 16, 16 );
    mapViewpointKernel<<< gridSize, blockSize>>>( in.Get(), out.Get(),
                                                  in.GetWidth(), in.GetHeight(),
                                                  in.GetPitch()  / sizeof(T),
                                                  out.GetPitch() / sizeof(T) );

    // sync host and stop computation timer
    checkCudaErrors( cudaDeviceSynchronize() );
}

template void runMapViewpoint<float>( GpuDepthMap<float> const& in, GpuDepthMap<float> &out );
template void runMapViewpoint<unsigned short>( GpuDepthMap<unsigned short> const& in, GpuDepthMap<unsigned short> &out );

//// toWorld ////

// Args:
//   depth_abs - the absolute depth from the kinect.
//   depth_proj - the projected depth.
__global__ void cam2WorldKernel( float *out,
                                 int w, int h,
                                 size_t out_pitch )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    float3 P_world = cam2World( (float2){x   , y   },
                                (float2){FX_RGB, FY_RGB}, (float2){CX_RGB, CY_RGB},
                                K1_RGB, K2_RGB, K3_RGB, K4_RGB, K5_RGB,
                                ALPHA_RGB );
    *(reinterpret_cast<float2*>(&out[ y * out_pitch + 2 * x ])) = (float2){P_world.x, P_world.y};
}

/*
 * \brief "undistort": Gives a mapping from {x, y} to {xn, yn, 1}
 * \param out Out is a (2*w, h) image containing normalised point coordinates (xn, yn)
 */
void cam2World( int w, int h, GpuDepthMap<float> &out )
{
    dim3 gridSize( (w + 16 - 1) / 16,
                   (h + 16 - 1) / 16 );
    dim3 blockSize( 16, 16 );
    cam2WorldKernel<<< gridSize, blockSize>>>( out.Get(),
                                               w, h,
                                               out.GetPitch() / sizeof(float) );

    // sync host and stop computation timer
    checkCudaErrors( cudaDeviceSynchronize() );
}
