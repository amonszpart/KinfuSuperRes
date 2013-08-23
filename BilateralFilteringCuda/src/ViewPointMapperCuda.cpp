#include "ViewPointMapperCuda.h"

#include "GpuDepthMap.hpp"
#include "AmCudaUtil.h"

#include <iostream>

// extern in ViewPointMapperCuda.cu
template<typename T>
extern void runMapViewpoint( GpuDepthMap<T> const& in, GpuDepthMap<T> &out );

void ViewPointMapperCuda::runViewpointMapping( float* const& in_data, float* out_data, int w, int h )
{
    if ( !out_data )
    {
        std::cerr << "ViewPointMapperCuda::runViewpointMapping: out_data pointer needs to be initalized!" << std::endl;
        return;
    }

    static GpuDepthMap<float> d_in;
    d_in.Create( DEPTH_MAP_TYPE_FLOAT, w, h );
    d_in.CopyDataIn( in_data );

    static GpuDepthMap<float> d_out;
    d_out.Create( DEPTH_MAP_TYPE_FLOAT, w, h );

    runMapViewpoint<float>( d_in, d_out );

    d_out.CopyDataOut( out_data );
}

void ViewPointMapperCuda::runViewpointMapping( cv::Mat const& in, cv::Mat &out )
{
    // copy in
    float *in_data = NULL;
    if ( in.type() == CV_16UC1 )
    {
        cv2Continuous32FC1<ushort,float>( in, in_data, 1.f /* / 10001.f*/ );
    }
    else if ( in.type() == CV_32FC1 )
    {
        cv2Continuous32FC1<float,float>( in, in_data, 1.f /* / 10001.f */ );
    }
    else
    {
        std::cerr << "ViewPointMapperCuda::runViewpointMapping: in.type needs to be 16UC1" << std::endl;
        return;
    }

    // prepare out
    float *out_data = new float[ in.cols * in.rows ];

    // work
    runViewpointMapping( in_data, out_data, in.cols, in.rows );

    // copy out
    if ( in.type() == CV_16UC1 )
    {
        cv::Mat fOut;
        continuous2Cv32FC1<float>( out_data, fOut, in.rows, in.cols, 1.f /*10001.f*/ );
        fOut.convertTo( out, CV_16UC1 );
    }
    else
    {
        continuous2Cv32FC1<float>( out_data, out, in.rows, in.cols, 1.f /*10001.f*/ );
    }

    // cleanup
    SAFE_DELETE_ARRAY( in_data );
    SAFE_DELETE_ARRAY( out_data );
}

void ViewPointMapperCuda::runViewpointMapping( unsigned short const* const& data, unsigned short* out, int w, int h )
{
    if ( !out )
    {
        std::cerr << "ViewPointMapperCuda::runViewpointMapping(): out needs to be initialized!" << std::endl;
        return;
    }

    // copy in
    const int size = w * h;
    float* fData = new float[ size ];
    for ( int i = 0; i < size; ++i )
    {
        fData[ i ] = static_cast<float>( data[i] );
    }

    // work
    runViewpointMapping( fData, fData, w, h );

    // copy out
    for ( int i = 0; i < size; ++i )
    {
        out[ i ] = static_cast<ushort>( round(fData[i]) );
    }

    // cleanup
    SAFE_DELETE_ARRAY( fData );
}

void ViewPointMapperCuda::runViewpointMapping( unsigned short *& data, int w, int h )
{
    runViewpointMapping( data, data, w, h );
}


// returns a float2 matrix of normalised 3D coordinates without the homogeneous part
extern void cam2World( int w, int h, GpuDepthMap<float> &out );
void ViewPointMapperCuda::runCam2World( int w, int h, float* out_data )
{
    // check input
    if ( !out_data )
    {
        std::cerr << "ViewPointMapperCuda::runCam2World(): out_data needs to be initialized!" << std::endl;
        return;
    }

    // input
    static GpuDepthMap<float> d_out;
    d_out.Create( DEPTH_MAP_TYPE_FLOAT, w * 2, h ); // two coordinates per pixel

    // work
    cam2World( w, h, d_out );

    // output
    d_out.CopyDataOut( out_data );
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

