#ifndef HOMOGRAPHYCALCULATOR_H
#define HOMOGRAPHYCALCULATOR_H

#include <opencv2/core/core.hpp>

class HomographyCalculator
{
    public:
        HomographyCalculator();
        void getHomography( cv::Mat img_object, cv::Mat img_scene );
};

#endif // HOMOGRAPHYCALCULATOR_H
