#ifndef YANGFILTERINGWRAPPER_H
#define YANGFILTERINGWRAPPER_H

#include <string>
#include "YangFiltering.h"
namespace am
{
    extern int runYangCleaned( cv::Mat &filteredDep16, std::string depPath, std::string imgPath, YangFilteringRunParams yangFilteringRunParams = YangFilteringRunParams(), std::string const& path = "./"  );
    extern int runYangCleaned( cv::Mat &filteredDep16, cv::Mat const& dep16, cv::Mat const& rgb8, YangFilteringRunParams yangFilteringRunParams = YangFilteringRunParams(), std::string const& path = "./" );

    extern int bruteRun( std::string depPath, std::string imgPath );
    extern int runYang(std::string depPath, std::string imgPath, YangFilteringRunParams yangFilteringRunParams = YangFilteringRunParams() );

} // end ns am

#endif // YANGFILTERINGWRAPPER_H
