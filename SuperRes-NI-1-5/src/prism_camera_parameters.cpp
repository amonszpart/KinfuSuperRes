#include "prism_camera_parameters.h"

void initPrismCamera( TCalibData &prismKinect )
{
    /// PrismKinect
    // Depth Intrinsics:
    // Focal Length: fc = [ 597.03208 597.22651 ] � [ 4.88666 4.84776 ]
    prismKinect.dep_intr.f = ( cv::Mat_<double>(1,2) << 597.03208, 597.22651 );

    // Principal point: cc = [ 311.39283 236.29659 ] � [ 6.80100 5.67643 ]
    prismKinect.dep_intr.c = ( cv::Mat_<double>(1,2) << 311.39283, 236.29659 );

    // Distortion: kc = [ -0.00820 0.01520 -0.00203 -0.00009 0.00000 ] � [ 0.01658 0.04823 0.00277 0.00328 0.00000 ]
    prismKinect.dep_intr.k = ( cv::Mat_<double>(1,5) << -0.00820, 0.01520, -0.00203, -0.00009, 0.00000 );

    // Pixel error: err = [ 0.30378 0.26392 ]

    // RGB Intrinsics:
    // Focal Length: fc_right = [ 1045.19670 1045.72682 ] � [ 7.05340 7.06538 ]
    prismKinect.rgb_intr.f = ( cv::Mat_<double>(1,2) << 1045.19670, 1045.72682 );
    // Principal point: cc_right = [ 639.19431 541.24726 ] � [ 6.06421 7.75099 ]
    prismKinect.rgb_intr.c = ( cv::Mat_<double>(1,2) << 639.19431, 541.24726 );
    // Distortion: kc_right = [ 0.17902 -0.35945 0.00852 -0.00205 0.00000 ] � [ 0.01569 0.03992 0.00267 0.00196 0.00000 ]
    prismKinect.rgb_intr.k = ( cv::Mat_<double>(1,5) << 0.17902, -0.35945, 0.00852, -0.00205, 0.00000 );

    // Extrinsic parameters (position of RGB camera wrt DEPTH camera): // Stereo_gui without intrinsic recalc
    // Rotation vector: om = [ 0.04869 0.00974 -0.00206 ] � [ 0.00423 0.00418 0.00254 ]
    prismKinect.R = (cv::Mat_<double>(3,3) <<   1.0000,    0.0023,    0.0097,
                                               -0.0018,    0.9988,   -0.0487,
                                               -0.0098,    0.0487,    0.9988 );
    // Translation vector: T = [ -33.25604 9.94095 -3.25410 ] � [ 2.43590 2.41533 1.47980 ]
    prismKinect.T = (cv::Mat_<double>(3,1) << -33.25604, 9.94095, -3.25410 );
 }

#if 0
// Focal Length: fc = [ 597.03208 597.22651 ] � [ 4.88666 4.84776 ]
prismKinect.dep_intr.f[0] = 597.03208;
prismKinect.dep_intr.f[1] = 597.22651;

// Principal point: cc = [ 311.39283 236.29659 ] � [ 6.80100 5.67643 ]
prismKinect.dep_intr.c[0] = 311.39283;
prismKinect.dep_intr.c[1] = 236.29659;

// Distortion: kc = [ -0.00820 0.01520 -0.00203 -0.00009 0.00000 ] � [ 0.01658 0.04823 0.00277 0.00328 0.00000 ]
prismKinect.dep_intr.k[0] = -0.00820;
prismKinect.dep_intr.k[1] = 0.01520;
prismKinect.dep_intr.k[2] =  -0.00203;
prismKinect.dep_intr.k[3] =  -0.00009;
prismKinect.dep_intr.k[4] = 0.00000;
#endif
