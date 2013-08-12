#include <iostream>
#include <pcl/console/parse.h>
#include <boost/filesystem.hpp>
#include <string>

#include "tsdf_viewer.h"
#include "my_pcd_viewer.h"
#include <eigen3/Eigen/Dense>

void printUsage()
{
    std::cout << "Usage:\n\tTSDFVis --in cloud.dat\n" << std::endl;
}

// --in /home/amonszpart/rec/testing/keyboard_cross_nomap_20130811_1828/cloud_tsdf_volume.dat
int main( char argc, char** argv )
{
    std::string tsdfFilePath;
    if (pcl::console::parse_argument (argc, argv, "--in", tsdfFilePath) < 0 )
    {
        printUsage();
        return 1;
    }

    //std::unique_ptr<am::MyPCDViewer> pcdViewer( new am::MyPCDViewer() );
    //pcdViewer->loadTsdf( pcdFilePath,  );
    //pcdViewer->run( pcdFilePath );


    // Load TSDF
    std::unique_ptr<am::TSDFViewer> tsdfViewer( new am::TSDFViewer() );
    tsdfViewer->loadTsdfFromFile( tsdfFilePath, true );

    tsdfViewer->dumpMesh();

    Eigen::Affine3f pose;
            pose.linear() <<  0.999154,  -0.0404336, -0.00822448,
                             0.0344457,    0.927101,   -0.373241,
                             0.0227151,    0.372638,    0.927706;
            pose.translation() << 1.63002,
                                  1.46289,
                                  0.227635;
    tsdfViewer->showGeneratedRayImage( tsdfViewer->kinfuVolume_ptr_, pose );
    tsdfViewer->showGeneratedDepth   ( tsdfViewer->kinfuVolume_ptr_, pose );
    tsdfViewer->spin();

    std::cout << "Hello Tsdf_vis" << std::endl;


}
