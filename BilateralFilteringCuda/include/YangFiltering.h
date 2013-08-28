#ifndef YANGFILTERING_H
#define YANGFILTERING_H

#include "BilateralFilterCuda.hpp"
#include "GpuDepthMap.hpp"
#include "GpuImage.h"

#include <opencv2/core/core.hpp>

struct YangFilteringRunParams
{
        YangFilteringRunParams()
            : L( 10 ),
              ETA( .5f ),
              MAXRES( 10001.f ),
              spatial_sigma( 1.1f ),
              range_sigma( .03f ),
              kernel_range( 4 ),
              cross_iterations( 1 ),
              fill_mode( FILL_ALL ),
              yang_iterations( 3 )
              {}

        int   L;
        float ETA;
        float MAXRES;
        float spatial_sigma;
        float range_sigma;
        int   kernel_range;
        int   cross_iterations;
        char  fill_mode;
        int   yang_iterations;
};

class YangFiltering
{
    public:
        int run( cv::Mat const& dep16, cv::Mat const& img8, cv::Mat &fDep, YangFilteringRunParams const params = YangFilteringRunParams(), std::string depPath = "./" );

    protected:
        GpuDepthMap<float>  d_fDep;
        GpuImage            d_guide;
        GpuDepthMap<float>  d_truncC2;
        GpuDepthMap<float>  d_filteredTruncC2s[2];
        GpuDepthMap<float>  d_crossFilterTemp;
        GpuDepthMap<float>  d_minDs;
        GpuDepthMap<float>  d_minC;
        GpuDepthMap<float>  d_minCm1;
        GpuDepthMap<float>  d_minCp1;
};

#endif // YANGFILTERING_H
