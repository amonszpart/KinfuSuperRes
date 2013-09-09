#ifndef VIEWPOINTMAPPERCUDA_H
#define VIEWPOINTMAPPERCUDA_H

#include <opencv2/core/core.hpp>
#include <eigen3/Eigen/Dense>

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

        /*
         *\brief Fills out_data with mapped homogeneous coordinates
         *\param out_data   preallocated float2 array of homogeneous coordinates of size [2*w x h], z == 1.f and not returned
         *\param w          image width
         *\param h          image height
         *\param fx         -1.f means fill from RGB header, 0.f means don't use
         */
        static void runCam2World( float *out_data, int w, int h, float fx = -1.f, float fy = -1.f, float cx = -1.f, float cy = -1.f,
                                  float k1 = -1.f, float k2 = -1.f, float p1 = -1.f, float p2 = -1.f, float k3 = -1.f, float alpha = -1.f );

        static void getIntrinsics( std::vector<float>& intrinsics, std::vector<float>& distortion_coeffs, INTRINSICS_CAMERA_ID camera );
        static void getIntrinsics( cv::Mat &intrinsics, cv::Mat &distortion_coeffs, INTRINSICS_CAMERA_ID camera, am::viewpoint_mapping::INTRINSICS_SCALE scale );
        static void getIntrinsics( Eigen::Matrix3f &intrinsics, cv::Mat &distortion_coeffs, INTRINSICS_CAMERA_ID camera, am::viewpoint_mapping::INTRINSICS_SCALE scale );
};

#endif // VIEWPOINTMAPPERCUDA_H
