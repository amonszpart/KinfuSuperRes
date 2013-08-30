#include <iostream>

#include "Recorder.h"
#include "../util/MaUtil.h"

#define SAMPLE_XML_PATH "../KinectNodesConfig.xml"

int main( int argc, char *argv[] )
{
    std::cout << "hello AMONI" << std::endl;

    std::string path = util::outputDirectoryNameWithTimestamp( "recording" );
    xnOSCreateDirectory( path.c_str() );
    path += "/amonirecorded.oni";
    std::cout << "path: " << path << std::endl;
    am::Recorder rtest( path, SAMPLE_XML_PATH );
    if ( XN_STATUS_OK == rtest.manualConfig(1280) )
    {
        rtest.setAltViewpoint( false );
        fflush(stdout);
        rtest.run( false );
    }

    return 0;
}
