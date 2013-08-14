#ifndef MYTHRUSTUTIL_H
#define MYTHRUSTUTIL_H

#include <opencv2/core/core.hpp>

class MyThrustUtil
{
    public:
        MyThrustUtil();
        static int runTest();
        static int squareDiff( cv::Mat const& a, float x, cv::Mat &out, float truncAt = -1.f );
        static int squareDiff( cv::Mat const& a, cv::Mat const& b, cv::Mat &out, float truncAt = -1.f);
        static int minMaskedCopy( const cv::Mat &Cprev ,
                                  const cv::Mat &C     ,
                                  const float   d      ,
                                        cv::Mat &minC  ,
                                        cv::Mat &minDs ,
                                        cv::Mat &minCm1,
                                        cv::Mat &minCp1 );
        static int subpixelRefine( cv::Mat const& minC  ,
                                   cv::Mat const& minCm1,
                                   cv::Mat const& minCp1,
                                   cv::Mat &minDs );
};

#endif // MYTHRUSTUTIL_H
