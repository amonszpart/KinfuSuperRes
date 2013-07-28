#ifndef PRISM_CAMERA_PARAMETERS_H
#define PRISM_CAMERA_PARAMETERS_H

#include <opencv2/core/core.hpp>
#include <memory>
#include "util/MaUtil.h"

struct TIntrinsics
{
    cv::Mat f, c, k;

    double fx() { return f.at<double>(0); };
    double fy() { return f.at<double>(1); };
    double cx() { return c.at<double>(0); };
    double cy() { return c.at<double>(1); };
};

struct TCalibData
{
    cv::Mat R, T;
    TIntrinsics rgb_intr;
    TIntrinsics dep_intr;
};

void initPrismCamera( TCalibData &prismKinect );

#endif // PRISM_CAMERA_PARAMETERS_H

/*double fx_rgb =  5.2921508098293293e+02;
double fy_rgb =  5.2556393630057437e+02;
double cx_rgb =  3.2894272028759258e+02;
double cy_rgb =  2.6748068171871557e+02;
double k1_rgb =  2.6451622333009589e-01;
double k2_rgb = -8.3990749424620825e-01;
double p1_rgb = -1.9922302173693159e-03;
double p2_rgb =  1.4371995932897616e-03;
double k3_rgb =  9.1192465078713847e-01;

double fx_d   =  5.9421434211923247e+02;
double fy_d   =  5.9104053696870778e+02;
double cx_d   =  3.3930780975300314e+02;
double cy_d   =  2.4273913761751615e+02;
double k1_d   = -2.6386489753128833e-01;
double k2_d   =  9.9966832163729757e-01;
double p1_d   = -7.6275862143610667e-04;
double p2_d   =  5.0350940090814270e-03;
double k3_d   = -1.3053628089976321e+00;*/
