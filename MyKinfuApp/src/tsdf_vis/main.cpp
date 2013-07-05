#include <iostream>
#include <pcl/console/parse.h>
#include <boost/filesystem.hpp>
#include <string>

//#include "tsdf_viewer.h"
#include "my_pcd_viewer.h"

void printUsage()
{
    std::cout << "Usage:\n\tTSDFVis --in cloud.pcd\n" << std::endl;
}

int main( char argc, char** argv )
{
    std::string pcdFilePath;
    if (pcl::console::parse_argument (argc, argv, "--in", pcdFilePath) < 0 )
    {
        printUsage();
        return 1;
    }

    std::unique_ptr<am::MyPCDViewer> pcdViewer( new am::MyPCDViewer() );
    pcdViewer->run( pcdFilePath );

    // Load TSDF
    //tsdfViewer->loadTsdfFromFile( tsdfFilePath, true );

    std::cout << "Hello Tsdf_vis" << std::endl;
}
