#ifndef VIEWPOINTMAPPERCUDA_H
#define VIEWPOINTMAPPERCUDA_H

#include <opencv2/core/core.hpp>

class ViewPointMapperCuda
{
    public:
        ViewPointMapperCuda();
        static void runMyCopyKernelTest( cv::Mat const& in, cv::Mat &out );
        static void runViewpointMapping( cv::Mat const& in, cv::Mat &out );
};

#endif // VIEWPOINTMAPPERCUDA_H
