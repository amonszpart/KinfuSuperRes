#ifndef AMCUDAUTIL_H
#define AMCUDAUTIL_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

void cv2Continuous8UC4( cv::Mat const& img, unsigned*& hImg, unsigned width, unsigned height, float alpha = 255.f / 10001.f );

template <typename inT, typename outT>
void cv2Continuous32FC1( cv::Mat const& in, outT*& out, float alpha = 1.f / 10001.f )
{
    if ( out ) delete [] out;
    out = new outT[ in.cols * in.rows ];

    unsigned offset = 0U;
    for ( unsigned y = 0; y < in.rows; ++y )
    {
        for ( unsigned x = 0; x < in.cols; ++x )
        {
            out[ offset ] = static_cast<outT>( static_cast<float>( in.at<inT>(y, x) ) * alpha );

            ++offset;
        }
    }
}

template <typename inT>
void continuous2Cv32FC1( float* const& in, cv::Mat & out, int h, int w, float alpha = 1.f )
{
    out.create( h, w, CV_32FC1 );
    unsigned offset = 0U;
    for ( unsigned y = 0; y < out.rows; ++y )
    {
        for ( unsigned x = 0; x < out.cols; ++x )
        {
            out.at<inT>(y, x) = in[ offset++ ] * alpha;
        }
    }
}

#endif // AMCUDAUTIL_H
