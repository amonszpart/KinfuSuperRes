#include <iostream>

#include "Recorder.h"

#define SAMPLE_XML_PATH "../KinectNodesConfig.xml"

int main( int argc, char *argv[] )
{
    std::cout << "hello AMONI" << std::endl;

    am::Recorder rtest( "./amonirecorded.oni", SAMPLE_XML_PATH );
    if ( XN_STATUS_OK == rtest.manualConfig(1280) )
    {
        rtest.setAltViewpoint( false );
        fflush(stdout);
        rtest.run( false );
    }

    return 0;
}


