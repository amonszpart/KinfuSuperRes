#include "tsdf_viewer.h"
#include "../kinfu/tools/kinfu_app.h"
#include "../kinfu/tools/kinfu_util.h"
#include <pcl/PolygonMesh.h>

namespace am
{

    TSDFViewer::TSDFViewer()
    {
        initCloudViewer();
    }

    void
    TSDFViewer::loadTsdfFromFile( std::string path, bool binary )
    {
        tsdf_volume_.load( path, binary );
        std::cout << "initing kinfuvolume with resolution: " << tsdf_volume_.gridResolution().transpose() << std::endl;

        pcl::gpu::TsdfVolume kinfuVolume( tsdf_volume_.gridResolution() );
        kinfuVolume.setSize( tsdf_volume_.header().volume_size );
        kinfuVolume.setTsdfTruncDist( 0.03f );
        std::cout << "copying kinfuvolume... " << std::endl;
        kinfu_.volume() = kinfuVolume;

        boost::shared_ptr<pcl::PolygonMesh> mesh_ptr;
        extractMeshFromVolume( kinfu_, mesh_ptr );

        //cloud_viewer_->removeAllPointClouds();
        //if (mesh_ptr)
        //  cloud_viewer_->addPolygonMesh(*mesh_ptr);

        pcl::io::savePLYFile( path + "_mesh.ply", *mesh_ptr );

    }

    void
    TSDFViewer::extractMeshFromVolume( const pcl::gpu::KinfuTracker &kinfu, boost::shared_ptr<pcl::PolygonMesh> mesh_ptr )
    {
        cout << "\nGetting mesh... " << flush;

        if ( !marching_cubes_ )
            marching_cubes_ = pcl::gpu::MarchingCubes::Ptr( new pcl::gpu::MarchingCubes() );

        DeviceArray<PointXYZ> triangles_buffer_device;
        DeviceArray<PointXYZ> triangles_device = marching_cubes_->run( kinfu.volume(), triangles_buffer_device );
        mesh_ptr = convertToMesh( triangles_device );

        cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
    }

    void
    TSDFViewer::initCloudViewer()
    {
        cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Scene Cloud Viewer") );

        cloud_viewer_->setBackgroundColor (0, 0, 0);
        cloud_viewer_->addCoordinateSystem (1.0);
        cloud_viewer_->initCameraParameters ();
        cloud_viewer_->setPosition (0, 500);
        cloud_viewer_->setSize (640, 480);
        cloud_viewer_->setCameraClipDistances (0.01, 10.01);

        cloud_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);
    }



} // ns am
