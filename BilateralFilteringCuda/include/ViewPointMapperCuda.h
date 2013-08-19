#ifndef VIEWPOINTMAPPERCUDA_H
#define VIEWPOINTMAPPERCUDA_H

#include <opencv2/core/core.hpp>

class ViewPointMapperCuda
{
    public:
        static void runViewpointMapping( cv::Mat                const& in     , cv::Mat       &  out                    );
        static void runViewpointMapping( unsigned short      *       & data   ,                            int w, int h );
        static void runViewpointMapping( float               *  const& in_data, float         *  out_data, int w, int h );
        static void runViewpointMapping(unsigned short const*  const& in_data, unsigned short * out_data, int w, int h );

        static void runMyCopyKernelTest( cv::Mat const& in, cv::Mat &out );
};

#endif // VIEWPOINTMAPPERCUDA_H
