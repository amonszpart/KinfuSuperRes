#ifndef AMKINFUTRACKER_H
#define AMKINFUTRACKER_H

#include <pcl/gpu/kinfu/kinfu.h>

class AMKinfuTracker : public pcl::gpu::KinfuTracker
{
        //bool operator() ( const DepthMap& depth_raw, Eigen::Affine3f *hint ) ;
};

#endif // AMKINFUTRACKER_H
