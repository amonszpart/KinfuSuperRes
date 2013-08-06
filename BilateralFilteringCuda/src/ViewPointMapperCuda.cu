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

#define HEIGHT 480
#define WIDTH 640

// The maximum observable depth, in meters.
#define MAX_DEPTH 10

// Constants for undoing the depth nonlinearity.
#define DN_W 0.3513e3f
#define DN_B 1.0925e3f

// comp_distortion_oulu.m rewrite from Bogouet toolbox
__device__ float2 CamDep2World( float2 xd,
                                float2 F,
                                float2 C,
                                float k1, // K1
                                float k2, // K2
                                float p1, // K3
                                float p2, // K4
                                float k3, // K5
                                float alpha = 0.f )
{
    // initial guess
    float2 xn = {
        (xd.x - C.x) / F.x,
        (xd.y - C.y) / F.y };

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

    return xn;
}

__device__ float2 World2CamRgb( float3 xw,
                                float2 F,
                                float2 C,
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
__global__ void mapViewpointKernel( const T* in,
                              T *out,
                              int w, int h,
                              size_t in_pitch,
                              size_t out_pitch )
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h)
        return;

    const float d = in[ y * w + x ] * 10001.f;

    //out[ y * w + x ] = px;

    float2 pn_left = {
        (x - CX_D) / FX_D,
        (y - CY_D) / FY_D };

    pn_left = CamDep2World( pn_left, (float2){FX_D, FY_D}, (float2){CX_D, CY_D},
                            K1_D, K2_D, K3_D, K4_D, K5_D, ALPHA_D );

    // ****************************************
    //   PROJECT THE DEPTH TO 3D WORLD POINTS
    // ****************************************
    float3 P_world_left = {
        pn_left.x * d,
        pn_left.y * d,
        d };

    // *******************************************
    //   Next, Rotate and translate the 3D points
    // *******************************************
    // R * [X; Y; Z] + T

    float3 P_world_right = {
        (R1 * P_world_left.x) + (R2 * P_world_left.y) + (R3 * P_world_left.z) + T1,
        (R4 * P_world_left.x) + (R5 * P_world_left.y) + (R6 * P_world_left.z) + T2,
        (R7 * P_world_left.x) + (R8 * P_world_left.y) + (R9 * P_world_left.z) + T3 };

    // *******************************************
    //   Project into the RGB coordinate frame.
    // *******************************************

    float2 p_right = World2CamRgb( P_world_right,
                                   (float2){FX_RGB, FY_RGB}, (float2){CX_RGB, CY_RGB},
                                   K1_RGB, K2_RGB, K3_RGB, K4_RGB, K5_RGB, ALPHA_RGB );

    // map to 0.f..1.f
    P_world_right.z /= 10001.f;
    // create int pointer
    int *pZ = (int*) &(P_world_right.z);

    // round to integer coordinates
    int2 int_p_right = { nearbyintf(p_right.x), nearbyintf(p_right.y) };

    // check boundaries
    if ( (int_p_right.x >= w) ||
         (int_p_right.y >= h) ||
         (int_p_right.x <  0) ||
         (int_p_right.y <  0)   ) return;

    float old = atomicCAS ( (int*)&(out[ int_p_right.y * w + int_p_right.x ]), 0, *pZ );
    if ( old != 0.f )
    {
        atomicMin ( (int*)&(out[ int_p_right.y * w + int_p_right.x ]), *pZ );
    }

#if 0
    // ************************************************
    // Finally, reassign the values in ROW MAJOR order.
    // ************************************************
    x = static_cast<int> (roundf(x_proj));
    y = static_cast<int> (roundf(y_proj));

    --x;
    --y;

    res_x[idx] = x;
    res_y[idx] = y;
#endif
}


template<typename T>
void runMapViewpoint( GpuDepthMap<T> const& in, GpuDepthMap<T> &out )
{
    cudaMemset( out.Get(), 0, out.GetWidth() * out.GetHeight() * sizeof(T) );
    checkCudaErrors( cudaDeviceSynchronize() );

    //runSetKernel2D( out.Get(), 22222.f, out.GetWidth(), out.GetHeight() );

    dim3 gridSize((in.GetWidth() + 16 - 1) / 16, (in.GetHeight() + 16 - 1) / 16);
    dim3 blockSize( 16, 16 );
    mapViewpointKernel<<< gridSize, blockSize>>>( in.Get(), out.Get(),
                                            in.GetWidth(), in.GetHeight(),
                                            in.GetPitch(), out.GetPitch() );

    // sync host and stop computation timer
    checkCudaErrors( cudaDeviceSynchronize() );
}

template void runMapViewpoint<float>( GpuDepthMap<float> const& in, GpuDepthMap<float> &out );
template void runMapViewpoint<unsigned short>( GpuDepthMap<unsigned short> const& in, GpuDepthMap<unsigned short> &out );
