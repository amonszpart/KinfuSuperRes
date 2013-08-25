#include "ViewPointMapperCuda.h"

#include "GpuDepthMap.hpp"
#include "AmCudaUtil.h"

#include <iostream>

/// viewpoint mapping with or without rgb lens distortion
// extern in ViewPointMapperCuda.cu
template<typename T>
extern void runMapViewpoint( GpuDepthMap<T> const& in, GpuDepthMap<T> &out, bool undistort );

void ViewPointMapperCuda::runViewpointMapping( float* const& in_data, float* out_data, int w, int h, bool undistort )
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

    runMapViewpoint<float>( d_in, d_out, undistort );

    d_out.CopyDataOut( out_data );
}

void ViewPointMapperCuda::runViewpointMapping( cv::Mat const& in, cv::Mat &out, bool undistort )
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
    runViewpointMapping( in_data, out_data, in.cols, in.rows, undistort );

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

void ViewPointMapperCuda::runViewpointMapping( unsigned short const* const& data, unsigned short* out, int w, int h, bool undistort )
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
    runViewpointMapping( fData, fData, w, h, undistort );

    // copy out
    for ( int i = 0; i < size; ++i )
    {
        out[ i ] = static_cast<ushort>( round(fData[i]) );
    }

    // cleanup
    SAFE_DELETE_ARRAY( fData );
}

void ViewPointMapperCuda::runViewpointMapping( unsigned short *& data, int w, int h, bool undistort )
{
    runViewpointMapping( data, data, w, h, undistort );
}

void ViewPointMapperCuda::undistortRgb( cv::Mat &undistortedRgb,
                                        cv::Mat const& rgb,
                                        am::viewpoint_mapping::INTRINSICS_SCALE in_scale,
                                        am::viewpoint_mapping::INTRINSICS_SCALE out_scale )
{
    // fetch current intrinsics
    cv::Mat intr_rgb, distr_rgb;
    ViewPointMapperCuda::getIntrinsics( intr_rgb, distr_rgb, RGB_CAMERA, in_scale );
    cv::Mat newIntrinsics;
    ViewPointMapperCuda::getIntrinsics( newIntrinsics, distr_rgb, RGB_CAMERA, out_scale );

    // resize for output
    newIntrinsics.at<float>( 0,1 ) = 0.f; // no skew please

    cv::undistort( rgb, undistortedRgb, intr_rgb, distr_rgb, newIntrinsics );
}

/// Intrinsics
extern void getIntrinsicsDEP( std::vector<float>& intrinsics, std::vector<float>& distortion_coeffs );
extern void getIntrinsicsRGB( std::vector<float>& intrinsics, std::vector<float>& distortion_coeffs );
void ViewPointMapperCuda::getIntrinsics( std::vector<float>& intrinsics, std::vector<float>& distortion_coeffs, INTRINSICS_CAMERA_ID camera )
{
    if      ( camera == DEP_CAMERA ) ::getIntrinsicsDEP( intrinsics, distortion_coeffs );
    else if ( camera == RGB_CAMERA ) ::getIntrinsicsRGB( intrinsics, distortion_coeffs );
    else std::cerr << "ViewPointMapperCuda::getIntrinsicsRGB(): unrecognized camera ID!!!" << std::endl;
}
void ViewPointMapperCuda::getIntrinsics( cv::Mat &intrinsics, cv::Mat &distortion_coeffs, INTRINSICS_CAMERA_ID camera, am::viewpoint_mapping::INTRINSICS_SCALE scale )
{
    // get
    std::vector<float> intr, distr;
    if      ( camera == DEP_CAMERA ) ::getIntrinsicsDEP( intr, distr );
    else if ( camera == RGB_CAMERA ) ::getIntrinsicsRGB( intr, distr );
    else std::cerr << "ViewPointMapperCuda::getIntrinsicsRGB(): unrecognized camera ID!!!" << std::endl;

    // prepare
    intrinsics       .create( 3,            3, CV_32FC1 );
    distortion_coeffs.create( 1,            5, CV_32FC1 );

    // get
    std::copy( intr .begin(), intr .end()      , intrinsics       .begin<float>() );
    std::copy( distr.begin(), distr.begin() + 5, distortion_coeffs.begin<float>() );

    // resize
    switch ( scale )
    {
        case am::viewpoint_mapping::INTR_RGB_640_480:
            // already there, calibration was done at this scale
            break;
        case am::viewpoint_mapping::INTR_RGB_1280_960:
            // multiply by two
            intrinsics *= 2.f;
            intrinsics.at<float>(2,2) = 1.f;
            break;
        case am::viewpoint_mapping::INTR_RGB_1280_1024:
            // multiply by two
            intrinsics.row(0) *= 2.f;
            intrinsics.row(1) *= am::viewpoint_mapping::_1024_DIV_480;
            break;
        default:
            std::cerr << "ViewPointMapperCuda::undistortRgb(): unrecognized input intrinsics scale!" << std::endl;
            break;
    }

    // alpha (skew)
    if ( distr.size() > 5 ) intrinsics.at<float>( 0,1 ) = distr[5];
    std::cout << "intr: " << intrinsics << std::endl;
}


/// returns a float2 matrix of normalised 3D coordinates without the homogeneous part
extern void cam2World( int w, int h, GpuDepthMap<float> &out ); // NOT TESTED!
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

/// CUDA testing
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


