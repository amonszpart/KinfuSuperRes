#include "my_pcd_viewer.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/pcd_io.h>

namespace am
{
    MyPCDViewer::MyPCDViewer()
    {}

    int
    MyPCDViewer::run( const std::string &file_name )
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud( new pcl::PointCloud<pcl::PointXYZI> );

        pcl::PCDReader pcd;
        if ( pcd.read(file_name, *cloud) < 0 )
            return (-1);

        //... populate cloud
        //pcl::io::loadPCDFile( file_name, *cloud );
        if (cloud->width * cloud->height == 0)
        {
            std::cerr << "[error: no points found!]\n";
            return (-1);
        }

        //pcl::visualization::CloudViewer viewer("Simple Cloud Viewer");
        boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer;
        viewer.reset (new pcl::visualization::PCLVisualizer ("PCD viewer"));

        int viewport;
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, viewport );
        viewer->setBackgroundColor (0, 0, 0, viewport);
        viewer->addText("Point Cloud1", 10, 10, "v1 text", viewport);
        //        viewer->addPointCloud<pcl::PointXYZ> (cloud, "sample cloud1", v1);
        viewer->addPointCloud<pcl::PointXYZI> (cloud, "sample cloud1", viewport);
        viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_IMMEDIATE_RENDERING, 1.0,  "sample cloud1" );


        while (!viewer->wasStopped())
        {
            //viewer.
            //boost::this_thread::sleep (boost::posix_time::microseconds (100000));
            viewer->spinOnce(1,true);
            boost::this_thread::sleep (boost::posix_time::microseconds (100000));
        }

        return EXIT_SUCCESS;
    }
} // ns am
