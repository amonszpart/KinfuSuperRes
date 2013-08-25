#ifndef UPSCALING_H
#define UPSCALING_H

#include "ProcessorWithIntrinsics.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>

namespace am
{

    class UpScaling : public ProcessorWithIntrinsics
    {
        public:
            UpScaling();
            UpScaling( Eigen::Matrix3f intrinsics );

            void
            run( std::string const& sPolygonPath, Eigen::Affine3f const& pose, cv::Mat const& rgb8, int img_id = -1, int p_cols = -1, int p_rows = -1, int argc = 0, char** argv = NULL );
    };

} // end ns am

#endif // UPSCALING_H
