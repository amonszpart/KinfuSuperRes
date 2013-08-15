#ifndef YANGFILTERING_H
#define YANGFILTERING_H

#include <opencv2/core/core.hpp>

class YangFiltering
{
    public:
        static int run(cv::Mat const& dep16, cv::Mat const& img8, cv::Mat &fDep, int iterations = 3);
};

#endif // YANGFILTERING_H
