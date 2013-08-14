#include "YangFiltering.h"

#include "BilateralFilterCuda.hpp"
#include "GpuDepthMap.hpp"
#include "AmCudaUtil.h"

#include <opencv2/highgui/highgui.hpp>

extern void squareDiff( GpuDepthMap<float> const& d_fDep, float d,
                        GpuDepthMap<float>      & d_C2  , float truncAt );
extern void minMaskedCopy( GpuDepthMap<float> const& C,
                           GpuDepthMap<float> const& Cprev ,
                           GpuDepthMap<float>      & minC  ,
                           GpuDepthMap<float>      & minCm1,
                           GpuDepthMap<float>      & minCp1,
                           GpuDepthMap<float>      & minDs ,
                           const float d );
extern void subpixelRefine( GpuDepthMap<float> const& minC  ,
                            GpuDepthMap<float> const& minCm1,
                            GpuDepthMap<float> const& minCp1,
                            GpuDepthMap<float> const& minDs ,
                            GpuDepthMap<float>      & fDep_next );

template <typename T>
extern T getMax( GpuDepthMap<T> const& img );

int YangFiltering::run( cv::Mat const& dep16, const cv::Mat &img8, cv::Mat &fDep )
{
    const int   L       = 10; // depth search range
    const float ETA     = .5f;
    const float ETA_L_2 = ETA*L*ETA*L;
    const float MAXRES  = 10001.f;

    const float spatial_sigma = 1.5f;
    const float range_sigma   = .03f;
    const int   kernel_range  = 3;
    const int   iterations    = 1;
    const char  fill_mode     = FILL_ALL;
    StopWatchInterface* kernel_timer = NULL;
    sdkCreateTimer( &kernel_timer );
    updateGaussian( spatial_sigma, kernel_range );

    float maxVal = MAXRES;

    /// parse input
    // fDep
    if ( dep16.type() == CV_16UC1 )
    {
        dep16.convertTo( fDep, CV_32FC1 );
    }
    else
    {
        dep16.copyTo( fDep );
    }

    float *fDepArr = NULL;
    toContinuousFloat( fDep, fDepArr );
    // move to device
    GpuDepthMap<float> d_fDep;
    d_fDep.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    d_fDep.CopyDataIn( fDepArr );

    // guide
    unsigned *guideArr = NULL;
    cv2Continuous8UC4( img8, guideArr, img8.cols, img8.rows, 1.f );
    GpuImage d_guide;
    d_guide.Create( IMAGE_TYPE_XRGB32, img8.cols, img8.rows );
    d_guide.CopyDataIn( guideArr );

    // device temporary memory
    GpuDepthMap<float> d_truncC2;
    d_truncC2.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    GpuDepthMap<float> d_filteredTruncC2s[2];
    d_filteredTruncC2s[0].Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    d_filteredTruncC2s[1].Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    uchar fc2_id = 0;

    GpuDepthMap<float> d_crossFilterTemp;
    d_crossFilterTemp.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );

    GpuDepthMap<float> d_minDs;
    d_minDs.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    GpuDepthMap<float> d_minC;
    d_minC.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    GpuDepthMap<float> d_minCm1;
    d_minCm1.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );
    GpuDepthMap<float> d_minCp1;
    d_minCp1.Create( DEPTH_MAP_TYPE_FLOAT, fDep.cols, fDep.rows );

    // filter cost slice
    const cudaExtent imageSizeCudaExtent = make_cudaExtent( d_truncC2.GetWidth(), d_truncC2.GetHeight(), 1 );

    for ( int it = 0; it < 3; ++it )
    {
        runSetKernel2D<float>( d_minC.Get(), maxVal*maxVal, d_minC.GetWidth(), d_minC.GetHeight() );
        runSetKernel2D<float>( d_minCm1.Get(), maxVal*maxVal, d_minCm1.GetWidth(), d_minCm1.GetHeight() );
        runSetKernel2D<float>( d_minCp1.Get(), maxVal*maxVal, d_minCp1.GetWidth(), d_minCp1.GetHeight() );

        int len = d_fDep.GetWidth() * d_fDep.GetHeight();
        float *tmp = new float[ len ];
        d_fDep.CopyDataOut( tmp );
        float mx = 0.f;
        for ( int i = 0; i < len; ++i )
            if ( tmp[i] > mx )
                mx = tmp[i];
        SAFE_DELETE_ARRAY( tmp );

        for ( int d = 0; d < std::min(mx + L + 1, MAXRES); d+=1 )
        {
            // debug
            std::cout << "d: " << d << " -> " << mx + L + 1 << std::endl;

            // calculate truncated cost
            squareDiff( /* in: */ d_fDep, /* curr depth: */ d, /* out: */ d_truncC2, /* truncAt: */ ETA_L_2 );


            crossBilateralFilterF<float>( /* out: */ d_filteredTruncC2s[fc2_id].Get(), d_filteredTruncC2s[fc2_id].GetPitch(),
                                          /* in:  */ d_truncC2.Get(), d_crossFilterTemp.Get(), d_truncC2.GetPitch(),
                                          d_guide.Get(), d_guide.GetPitch(),
                                          imageSizeCudaExtent,
                                          range_sigma, kernel_range, iterations, fill_mode,
                                          kernel_timer );

            // track minimums
            minMaskedCopy( /* curr C: */ d_filteredTruncC2s[ fc2_id       ],
                           /* prev C: */ d_filteredTruncC2s[ (fc2_id+1)%2 ],
                                         d_minC, d_minCm1, d_minCp1, d_minDs,
                                         d );
            // swap filter output targets
            fc2_id = (fc2_id+1) % 2;
        }

        // refine minDs based on C(d_min), C(d_min-1), C(d_min+1)
        subpixelRefine( d_minC, d_minCm1, d_minCp1, d_minDs, d_fDep );

        // debug - copy out
        d_fDep.CopyDataOut( fDepArr );
        fromContinuousFloat( fDepArr, fDep );

        char title[255];
        sprintf( title, "iteration%d.png", it );

        // out
        cv::Mat dep_out;
        fDep.convertTo( dep_out, CV_16UC1 );

        // show
        cv::imshow( title, fDep / 10001.f );

        // write
        std::vector<int> imwrite_params;
        imwrite_params.push_back( cv::IMWRITE_PNG_COMPRESSION );
        imwrite_params.push_back( 0 );
        cv::imwrite( title, dep_out, imwrite_params );

        // float(in)
        cv::Mat ftmp;
        dep16.convertTo( ftmp, CV_32FC1 );

        // diff
        cv::Mat diff;
        cv::absdiff( ftmp, fDep, diff );
        // show
        cv::imshow( "diff", diff/10001.f );
        // write
        sprintf( title, "diff_iteration%d.png", it );
        cv::Mat diffUC16;
        diff.convertTo( diffUC16, CV_16UC1 );
        cv::imwrite( title, diffUC16, imwrite_params );

        cv::waitKey(20);
    }

    // copy out
    d_fDep.CopyDataOut( fDepArr );
    fromContinuousFloat( fDepArr, fDep );

    // cleanup
    SAFE_DELETE_ARRAY( fDepArr );
    SAFE_DELETE_ARRAY( guideArr );

    return 0;
}

