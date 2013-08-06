#include "ViewPointMapperCuda.h"

#include "GpuDepthMap.h"
#include "AmCudaUtil.h"

#include <iostream>

ViewPointMapperCuda::ViewPointMapperCuda()
{
}

template <typename T>
extern void runCopyKernel2D( T *in , unsigned w_in , unsigned h_in , size_t pitch_in,
                             T *out, size_t pitch_out );

void ViewPointMapperCuda::runMyCopyKernelTest( cv::Mat const& in, cv::Mat &out )
{
    GpuDepthMap<float> d_in;
    {
        d_in.Create( DEPTH_MAP_TYPE_FLOAT, in.cols, in.rows );
        float *tmp = NULL;
        cv2Continuous32FC1<ushort>( in, tmp, 10001.f );
        d_in.CopyDataIn( tmp );
        delete [] tmp;
        tmp = NULL;
    }

    GpuDepthMap<float> d_out;
    d_out.Create( DEPTH_MAP_TYPE_FLOAT, in.cols, in.rows );

    runCopyKernel2D<float>( d_in.Get() , d_in.GetWidth() , d_in.GetHeight() , d_in.GetPitch(),
                            d_out.Get(), d_out.GetPitch() );
    // copy out
    {
        float *tmp = new float[ in.cols * in.rows ];
        d_out.CopyDataOut( tmp );
        continuous2Cv32FC1<float>( tmp, out, in.rows, in.cols );
        delete [] tmp; tmp = NULL;
    }
}


template<typename T>
extern void runMapViewpoint( GpuDepthMap<T> const& in, GpuDepthMap<T> &out );

void ViewPointMapperCuda::runViewpointMapping( cv::Mat const& in, cv::Mat &out )
{
    static GpuDepthMap<float> d_in;
    {
        d_in.Create( DEPTH_MAP_TYPE_FLOAT, in.cols, in.rows );
        float *tmp = NULL;
        cv2Continuous32FC1<ushort,float>( in, tmp, 1.f / 10001.f );
        d_in.CopyDataIn( tmp );
        delete [] tmp;
        tmp = NULL;
    }

    static GpuDepthMap<float> d_out;
    d_out.Create( DEPTH_MAP_TYPE_FLOAT, in.cols, in.rows );

    runMapViewpoint<float>( d_in, d_out );

    // copy out
    {
        float *tmp = new float[ in.cols * in.rows ];
        d_out.CopyDataOut( tmp );
        cv::Mat cvTmp;
        continuous2Cv32FC1<float>( tmp, cvTmp, in.rows, in.cols, 10001.f );
        delete [] tmp; tmp = NULL;
        cvTmp.convertTo( out, CV_16UC1 );
    }
}
