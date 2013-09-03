#ifndef AMUTIL_H
#define AMUTIL_H

#include "opencv2/core/core.hpp"
#include <string>

namespace am
{
    namespace util
    {
        extern int
        savePFM( cv::Mat const& imgF, std::string path, float scale = -1.f );

        extern int
        loadPFM( cv::Mat & imgF, std::string path );

        namespace cv
        {
            extern void
            unsignedIntToFloat( ::cv::Mat & img32FC1, ::cv::Mat const& img32UC1 );

            extern int
            imread( /* out: */ ::cv::Mat &mat, /*  in: */ std::string const& path );
        }

    } // end ns util
} // end ns am

#endif // AMUTIL_H
