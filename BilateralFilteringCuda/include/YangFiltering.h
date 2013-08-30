#ifndef YANGFILTERING_H
#define YANGFILTERING_H

#include "BilateralFilterCuda.hpp"
#include "GpuDepthMap.hpp"
#include "GpuImage.h"

#include <opencv2/core/core.hpp>

struct YangFilteringRunParams
{
        YangFilteringRunParams()
            : spatial_sigma( 1.1f ),
              range_sigma( .03f ),
              kernel_range( 4 ),
              cross_iterations( 1 ),
              fill_mode( FILL_ALL ),
              L( 10 ),
              ETA( .5f ),
              MAXRES( 10001.f ),
              d_step( 1.f ),
              yang_iterations( 3 )
        {}
        // crossFilter params
        float spatial_sigma;    // x,y gaussian falloff
        float range_sigma;      // rgb gaussian falloff
        int   kernel_range;     // gaussian kernel radius
        int   cross_iterations; // iterations to crossbilateral filter each cost volume
        char  fill_mode;        // fill only zeros, zeros first, or all
        // yangFilter params
        int   L;                // lookuprange
        float ETA;              // lookuprange multiplier
        float MAXRES;           // maximum depth to look at (i.e. 255.f, or 10001.f)
        float d_step;           // search step
        int   yang_iterations;  // overall iterations
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
