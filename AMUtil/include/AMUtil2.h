#ifndef AMUTIL_H
#define AMUTIL_H

#include "opencv2/core/core.hpp"
#include <string>

namespace am
{
    namespace util
    {
        extern void
        savePFM( cv::Mat const& imgF, std::string path, float scale = -1.f );

        extern void
        loadPFM( cv::Mat & imgF, std::string path );

        namespace cv
        {
            extern void
            unsignedIntToFloat( ::cv::Mat & img32FC1, ::cv::Mat const& img32UC1 );
        }

    } // end ns util
} // end ns am

#endif // AMUTIL_H
