#ifndef PROCESSORWITHINTRINSICS_H
#define PROCESSORWITHINTRINSICS_H

#include <eigen3/Eigen/Dense>

namespace am
{
    class ProcessorWithIntrinsics
    {
        public:
            ProcessorWithIntrinsics()
            {
                // default from KinfuTracker
                intrinsics_ << 525.f, 0    , 319.5f,
                                0   , 525.f, 239.5f,
                                0   , 0    ,    1.f;
            }


            ProcessorWithIntrinsics( Eigen::Matrix3f p_intrinsics )
            {
                intrinsics_ = p_intrinsics;
            }

        protected:
            Eigen::Matrix3f intrinsics_;
    };
}

#endif // PROCESSORWITHINTRINSICS_H
