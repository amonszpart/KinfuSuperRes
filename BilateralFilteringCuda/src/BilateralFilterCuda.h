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
double crossBilateralFilterRGBA( unsigned *dDest,
                                 unsigned *dImage, unsigned *dTemp, unsigned pitch,
                                 unsigned *dGuide, unsigned guidePitch,
                                 int width, int height,
                                 float e_d, int radius, int iterations,
                                 StopWatchInterface *timer
                                 );
extern "C"
double crossBilateralFilterF( float *dDest,
                              float *dImage, float *dTemp, uint pitch,
                              unsigned *dGuide, unsigned guidePitch,
                              int width, int height,
                              float e_d, int radius, int iterations,
                              StopWatchInterface *timer
                              );

class BilateralFilterCuda
{
        float               m_gaussian_delta;
        float               m_euclidean_delta;
        int                 m_filter_radius;
        StopWatchInterface  *m_kernel_timer;
        int                 m_iterations;
    public:
        BilateralFilterCuda();
        void runBilateralFiltering( cv::Mat const& in, cv::Mat const &guide, cv::Mat &out );
};

#endif //__BILATERAL_FILTER_CUDA_H
