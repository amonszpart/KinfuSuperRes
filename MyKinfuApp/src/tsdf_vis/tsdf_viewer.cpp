#include "tsdf_viewer.h"
#include "../kinfu/tools/kinfu_app.h"
#include "../kinfu/tools/kinfu_util.h"
#include <pcl/PolygonMesh.h>
#include <vector_types.h> // cuda-5.0



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
        std::vector<short2> tsdf( tsdf_volume_.header().getVolumeSize() );

#       pragma omp parallel for
        for(int i = 0; i < (int) tsdf.size(); ++i)
        {
            tsdf[i].x = tsdf_volume_.volume()[i] * device::DIVISOR;
            tsdf[i].y = tsdf_volume_.weights()[i];

            /*short2 elem = *reinterpret_cast<short2*>(&tsdf[i]);
              tsdf[i] = (float)(elem.x)/device::DIVISOR;
              weights[i] = (short)(elem.y);*/
        }


        // init kinfuVolume
        Eigen::Vector3i volSize = tsdf_volume_.gridResolution();
        pcl::gpu::TsdfVolume::Ptr kinfuVolume( new pcl::gpu::TsdfVolume(tsdf_volume_.gridResolution()) );
        kinfuVolume->setSize( tsdf_volume_.header().volume_size );
        kinfuVolume->setTsdfTruncDist( 0.03f );
        kinfuVolume->reset();

        // tsdf_volume_ --> kinfuVolume
        std::cout << "copying tsdf_volume_ --> kinfuVolume... ";
        //volume_.create (volume_y * volume_z, volume_x);
        kinfuVolume->data().upload( &tsdf[0], volSize(0) * sizeof(int), volSize(1)*volSize(2), volSize(0) );
        std::cout << "OK" << std::endl;

        /*int volumeSize = volume_.cols() * volume_.rows();
          tsdf.resize (volumeSize);
          weights.resize (volumeSize);
          volume_.download(&tsdf[0], volume_.cols() * sizeof(int));

        #pragma omp parallel for
          for(int i = 0; i < (int) tsdf.size(); ++i)
          {
            short2 elem = *reinterpret_cast<short2*>(&tsdf[i]);
            tsdf[i] = (float)(elem.x)/device::DIVISOR;
            weights[i] = (short)(elem.y);
          }
        }*/
        //file.write ((char*) &(volume_->at(0)), volume_->size()*sizeof(VoxelT));
        //file.write ((char*) &(weights_->at(0)), weights_->size()*sizeof(WeightT));

        // kinfuVolume --> kinfu_.volume()
        std::cout << "copying kinfuVolume --> kinfu_.volume()... ";
        kinfu_.volume() = *kinfuVolume;
        std::cout << "OK" << std::endl;

        // extract mesh
        boost::shared_ptr<pcl::PolygonMesh> mesh_ptr;
        extractMeshFromVolume( kinfuVolume, mesh_ptr );

        //cloud_viewer_->removeAllPointClouds();
        //if (mesh_ptr)
        //  cloud_viewer_->addPolygonMesh(*mesh_ptr);

        // save mesh
        pcl::io::savePLYFile( path + "_mesh.ply", *mesh_ptr );
        std::cout << "saved mesh in " << path + "_mesh.ply" << std::endl;

    }

    void
    TSDFViewer::extractMeshFromVolume( const pcl::gpu::TsdfVolume::Ptr volume, boost::shared_ptr<pcl::PolygonMesh>& mesh_ptr )
    {
        cout << "\nGetting mesh... " << flush;

        if ( !marching_cubes_ )
            marching_cubes_ = pcl::gpu::MarchingCubes::Ptr( new pcl::gpu::MarchingCubes() );

        DeviceArray<PointXYZ> triangles_buffer_device;
        DeviceArray<PointXYZ> triangles_device = marching_cubes_->run( *volume, triangles_buffer_device );
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
