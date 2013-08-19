#ifndef YANGFILTERINGWRAPPER_H
#define YANGFILTERINGWRAPPER_H

#include <string>
#include "YangFiltering.h"

int bruteRun( std::string depPath, std::string imgPath );
int runYang(std::string depPath, std::string imgPath, YangFilteringRunParams yangFilteringRunParams = YangFilteringRunParams() );

#endif // YANGFILTERINGWRAPPER_H
