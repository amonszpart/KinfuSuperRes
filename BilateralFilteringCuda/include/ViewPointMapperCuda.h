#ifndef VIEWPOINTMAPPERCUDA_H
#define VIEWPOINTMAPPERCUDA_H

#include <opencv2/core/core.hpp>

enum INTRINSICS_CAMERA_ID
{
    DEP_CAMERA,
    RGB_CAMERA,
    LEFT_CAMERA = DEP_CAMERA,
    RIGHT_CAMERA = RGB_CAMERA
};

namespace am
{
    namespace viewpoint_mapping
    {
        enum INTRINSICS_SCALE { INTR_RGB_640_480, INTR_RGB_1280_960, INTR_RGB_1280_1024 };
        const float _1024_DIV_480 = 1024.f/480.f;
    }
}

class ViewPointMapperCuda
{
    public:
        static void runViewpointMapping( cv::Mat                const& in     , cv::Mat        & out                   , bool undistort = false );
        static void runViewpointMapping( unsigned short      *       & data   ,                            int w, int h, bool undistort = false );
        static void runViewpointMapping( float               *  const& in_data, float          * out_data, int w, int h, bool undistort = false );
        static void runViewpointMapping( unsigned short const*  const& in_data, unsigned short * out_data, int w, int h, bool undistort = false );

        static void undistortRgb( cv::Mat &undistortedRgb,
                                  cv::Mat const& rgb,
                                  am::viewpoint_mapping::INTRINSICS_SCALE in_scale,
                                  am::viewpoint_mapping::INTRINSICS_SCALE out_scale );

        static void runMyCopyKernelTest( cv::Mat const& in, cv::Mat &out );

        static void runCam2World( int w, int h, float* out_data );

        static void getIntrinsics( std::vector<float>& intrinsics, std::vector<float>& distortion_coeffs, INTRINSICS_CAMERA_ID camera );
        static void getIntrinsics( cv::Mat &intrinsics, cv::Mat &distortion_coeffs, INTRINSICS_CAMERA_ID camera, am::viewpoint_mapping::INTRINSICS_SCALE scale );
};

#endif // VIEWPOINTMAPPERCUDA_H
