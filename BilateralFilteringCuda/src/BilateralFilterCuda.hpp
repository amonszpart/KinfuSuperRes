#ifndef __BILATERAL_FILTER_CUDA_H
#define __BILATERAL_FILTER_CUDA_H
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

#include "GpuImage.h"
#include "GpuDepthMap.h"

// CUDA utilities and system includes
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
// CUDA device initialization helper functions
#include <helper_cuda.h>
// CUDA SDK Helper functions
#include <helper_functions.h>

#include "AmCudaUtil.h"
#include <opencv2/core/core.hpp>

class StopWatchInterface;

/*
 *  Because a 2D gaussian mask is symmetry in row and column,
 *  here only generate a 1D mask, and use the product by row
 *  and column index later.
 *
 *  1D gaussian distribution :
 *      g(x, d) -- C * exp(-x^2/d^2), C is a constant amplifier
 *
 *  @param og       Output gaussian array in global memory
 *  @param delta    The 2nd parameter 'd' in the above function
 *  @param radius   Half of the filter size (total filter size = 2 * radius + 1)
*/
extern "C"
void updateGaussian( float delta, int radius );

/*
 * @brief               Perform 2D bilateral filter on image using CUDA
 * @param d_dest        Pointer to destination image in device memory
 * @param width         Image width
 * @param height        Image height
 * @param e_d           Euclidean delta
 * @param radius        Filter radius
 * @param iterations    Number of iterations
*/
extern "C"
double bilateralFilterRGBA( unsigned *dDest,
                            int width, int height,
                            float e_d, int radius, int iterations,
                            StopWatchInterface *timer,
                            unsigned* dImage, unsigned* dTemp, unsigned pitch );

extern "C"
double bilateralFilterF( float *dDest,
                         int width, int height,
                         float e_d, int radius, int iterations,
                         StopWatchInterface *timer,
                         float* dImage, float* dTemp, uint pitch );

extern "C"
double crossBilateralFilterRGBA( unsigned *dDest,
                                 unsigned *dImage, unsigned *dTemp, unsigned pitch,
                                 unsigned *dGuide, unsigned guidePitch,
                                 int width, int height,
                                 float e_d, int radius, int iterations,
                                 StopWatchInterface *timer
                                 );
template <typename T>
extern double crossBilateralFilterF( T *dDest,
                                     T *dImage, T *dTemp, uint pitch,
                                     unsigned *dGuide, unsigned guidePitch,
                                     int width, int height,
                                     float e_d, int radius, int iterations,
                                     StopWatchInterface *timer
                                     );

template <typename T>
class BilateralFilterCuda
{
    public:
        BilateralFilterCuda();

        void runBilateralFiltering( cv::Mat const& in, cv::Mat const &guide, cv::Mat &out,
                                    float gaussian_delta = -1.f, float euclidian_delta = -.1f, int filter_radius = -2 );
        void runBilateralFiltering( ushort* const& in, unsigned* const& guide, ushort*& out,
                                    unsigned width, unsigned height,
                                    float gaussian_delta, float eucledian_delta, int filter_radius );
        void runBilateralFiltering( T* const& in, unsigned* const& guide, T*& out,
                                    unsigned width, unsigned height,
                                    float gaussian_delta, float euclidian_delta, int filter_radius );


        void setGaussianParameters( float gaussian_delta, int filter_radius );

    private:
        float               m_gaussian_delta;
        float               m_euclidean_delta;
        int                 m_filter_radius;
        StopWatchInterface  *m_kernel_timer;
        int                 m_iterations;

        GpuDepthMap<T>         m_dDep16;
        GpuDepthMap<T>         m_dTemp, m_dFiltered;
        GpuImage            m_dGuide;
};

template <typename T>
BilateralFilterCuda<T>::BilateralFilterCuda()
    : m_gaussian_delta(2.f), m_euclidean_delta( .1f ), m_filter_radius(2), m_iterations(1)
{
    this->setGaussianParameters( m_gaussian_delta, m_filter_radius );
    sdkCreateTimer( &m_kernel_timer );
}

template <typename T>
void BilateralFilterCuda<T>::setGaussianParameters( float gaussian_delta, int filter_radius )
{
    m_gaussian_delta = gaussian_delta;
    m_filter_radius  = filter_radius;
    updateGaussian( m_gaussian_delta, m_filter_radius );
}


// U16 -> float -> run -> float -> U16
template <typename T>
void BilateralFilterCuda<T>::runBilateralFiltering( cv::Mat const& in, cv::Mat const &guide, cv::Mat &out,
                                                    float gaussian_delta, float eucledian_delta, int filter_radius )
{
    // empty - check input content
    if ( in.empty() )
        return;

    // type - check input type
    if ( in.type() != CV_16UC1 )
    {
        std::cerr << "BilateralFilterCuda::runBilateralFiltering input is expected to be CV_16UC1 with max 10001.f...exiting..." << std::endl;
        return;
    }

    // in
    T *inData = NULL;
    cv2Continuous32FC1<ushort,T>( in, inData, 1.f / 10001.f );

    // guide
    unsigned *guideData = NULL;
    if ( !guide.empty() )
    {
        cv2Continuous8UC4( guide, guideData, in.cols, in.rows, 1.f );
    }

    // out
    T *outData = new T[ in.cols * in.rows ];

    // work
    runBilateralFiltering( inData, guideData, outData, in.cols, in.rows, gaussian_delta, eucledian_delta, filter_radius );

    // out
    {
        cv::Mat cvTmp;
        continuous2Cv32FC1<T>( outData, cvTmp, in.rows, in.cols, 10001.f );

        cvTmp.convertTo( out, CV_16UC1 );
    }

    // cleanup
    SAFE_DELETE_ARRAY( inData  );
    SAFE_DELETE_ARRAY( outData );
    SAFE_FREE( guideData );
}

// ushort* -> float -> run -> float -> ushort*
template <typename T>
void BilateralFilterCuda<T>::runBilateralFiltering( ushort* const& in, unsigned* const& guide, ushort*& out,
                                                    unsigned width, unsigned height,
                                                    float gaussian_delta, float eucledian_delta, int filter_radius )
{
    const int size = width * height;

    // in - rescale
    float* tmpIn = new float[ size ];
    for ( int i = 0; i < size; ++i)
    {
        tmpIn[i] = (float)in[i] / 10001.f;

        // debug
        if ( in[i] > 255 )
            std::cout << "in ok..." << in[i] << std::endl;
    }

    // out
    T* tmpOut = new T[ size ];

    runBilateralFiltering( in, guide, tmpOut, width, height, gaussian_delta, eucledian_delta, filter_radius );

    for ( int i = 0; i < size; ++i )
    {
        out[i] = static_cast<ushort>(roundf(tmpOut[i] * 10001.f));
    }

    SAFE_DELETE_ARRAY( tmpIn  );
    SAFE_DELETE_ARRAY( tmpOut );
}

/*
 * Assumes everything is allocated from before
 **/
template <typename T>
void BilateralFilterCuda<T>::runBilateralFiltering( T* const& in, unsigned* const& guide, T*& out,
                                                    unsigned width, unsigned height,
                                                    float gaussian_delta, float eucledian_delta, int filter_radius )
{

    // params - drop not-meaningful input parameters
    if ( gaussian_delta  <  0.f ) gaussian_delta    = m_gaussian_delta;
    if ( filter_radius   <  0.f ) filter_radius     = m_filter_radius;
    if ( (gaussian_delta != m_gaussian_delta) || (filter_radius != m_filter_radius) )
    {
        this->setGaussianParameters( gaussian_delta, filter_radius );
    }
    if ( eucledian_delta >= 0.f ) m_euclidean_delta = eucledian_delta;

    // dDep16 (input), dTemp
    {
        m_dDep16.Create( DEPTH_MAP_TYPE_FLOAT, width, height );
        m_dTemp .Create( DEPTH_MAP_TYPE_FLOAT, width, height );

        m_dDep16.CopyDataIn( in );
    }

    // dGuide
    if ( guide )
    {
        m_dGuide.Create( IMAGE_TYPE_XRGB32, width, height);

        m_dGuide.CopyDataIn( guide );
    }

    // dFiltered (output)
    m_dFiltered.Create( DEPTH_MAP_TYPE_FLOAT, width, height );

    // work
    if ( !guide )
    {
        bilateralFilterF( m_dFiltered.Get(),
                          m_dDep16.GetWidth(), m_dDep16.GetHeight(),
                          m_euclidean_delta, m_filter_radius, m_iterations,
                          m_kernel_timer,
                          m_dDep16.Get(), m_dTemp.Get(), m_dDep16.GetPitch() );
    }
    else
    {
        crossBilateralFilterF<T>( m_dFiltered.Get(),
                                  m_dDep16.Get(), m_dTemp.Get(), m_dDep16.GetPitch(),
                                  m_dGuide.Get(), m_dGuide.GetPitch(),
                                  m_dDep16.GetWidth(), m_dDep16.GetHeight(),
                                  m_euclidean_delta, m_filter_radius, m_iterations, m_kernel_timer );
    }

    // copy output from device
    {
        //T *tmp = new T[ in.cols * in.rows ];
        m_dFiltered.CopyDataOut( out );
    }

}


#endif //__BILATERAL_FILTER_CUDA_H
