#include "tsdf_viewer.h"
#include "../kinfu/tools/kinfu_app.h"
#include "../kinfu/tools/kinfu_util.h"
#include <pcl/PolygonMesh.h>
#include <vector_types.h> // cuda-5.0

namespace am
{

    TSDFViewer::TSDFViewer()
    {
        const int rows = 480;
        const int cols = 640;

        //initCloudViewer( rows, cols );
        initRayViewer( rows, cols );
        initDepthViewer( rows, cols );
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
        std::cout << kinfuVolume_ptr_.use_count() << std::endl;
        //pcl::gpu::TsdfVolume::Ptr tmpVolume( new pcl::gpu::TsdfVolume(tsdf_volume_.gridResolution()) );
        //kinfuVolume = tmpVolume;
        //tmpVolume.reset();
        kinfuVolume_ptr_ = pcl::gpu::TsdfVolume::Ptr( new pcl::gpu::TsdfVolume(tsdf_volume_.gridResolution()) );
        kinfuVolume_ptr_->setSize( tsdf_volume_.header().volume_size );
        kinfuVolume_ptr_->setTsdfTruncDist( 0.03f );
        kinfuVolume_ptr_->reset();

        // tsdf_volume_ --> kinfuVolume
        std::cout << "copying tsdf_volume_ --> kinfuVolume... ";
        //volume_.create (volume_y * volume_z, volume_x);
        kinfuVolume_ptr_->data().upload( &tsdf[0], volSize(0) * sizeof(int), volSize(1)*volSize(2), volSize(0) );
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
        //std::cout << "copying kinfuVolume --> kinfu_.volume()... ";
        //kinfu_.volume() = *kinfuVolume;
        //std::cout << "OK" << std::endl;

        // extract mesh
        //boost::shared_ptr<pcl::PolygonMesh> mesh_ptr;
        //extractMeshFromVolume( kinfuVolume, mesh_ptr );

        //cloud_viewer_->removeAllPointClouds();
        //if (mesh_ptr)
        //  cloud_viewer_->addPolygonMesh(*mesh_ptr);

        // save mesh
        //if ( mesh_ptr.get() != NULL )
        //{
        //    pcl::io::savePLYFile( path + "_mesh.ply", *mesh_ptr );
        //    std::cout << "saved mesh in " << path + "_mesh.ply" << std::endl;
        //}
    }

    void
    TSDFViewer::dumpMesh( std::string path )
    {

        // extract mesh
        boost::shared_ptr<pcl::PolygonMesh> mesh_ptr;
        extractMeshFromVolume( kinfuVolume_ptr_, mesh_ptr );

        // save mesh
        if ( mesh_ptr.get() != NULL )
        {
            pcl::io::savePLYFile( path + "_mesh.ply", *mesh_ptr );
            std::cout << "saved mesh in " << path + "_mesh.ply" << std::endl;
        }
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
    TSDFViewer::initCloudViewer( int rows, int cols )
    {
        cloud_viewer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Scene Cloud Viewer") );

        cloud_viewer_->setBackgroundColor (0, 0, 0);
        cloud_viewer_->addCoordinateSystem (1.0);
        cloud_viewer_->initCameraParameters ();
        cloud_viewer_->setPosition (0, 500);
        cloud_viewer_->setSize (cols, rows);
        cloud_viewer_->setCameraClipDistances (0.01, 10.01);

        cloud_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);
    }

    void
    TSDFViewer::initRayViewer( int rows, int cols )
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 0);


        if ( !raycaster_ptr_ )
        {
            raycaster_ptr_ = RayCaster::Ptr( new RayCaster( rows, cols ) );
            raycaster_ptr_->setIntrinsics( 587.97535, 587.81351, 314.51750, 240.80013 );
        }
        //raycaster_ptr_->
    }

    void
    TSDFViewer::initDepthViewer( int rows, int cols )
    {
        viewerDepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerDepth_->setWindowTitle ("Depth from ray tracing");
        viewerDepth_->setPosition (640, 0);


        if ( !raycaster_ptr_ )
        {
            raycaster_ptr_ = RayCaster::Ptr( new RayCaster( rows, cols ) );
            raycaster_ptr_->setIntrinsics( 587.97535, 587.81351, 314.51750, 240.80013 );
        }
        //raycaster_ptr_->
    }

    void
    TSDFViewer::showGeneratedDepth ( pcl::gpu::TsdfVolume::Ptr const& volume, const Eigen::Affine3f& pose )
    {
        //pose.linee
        std::cout << "Running raycaster...";
        raycaster_ptr_->run( *this->kinfuVolume_ptr_.get(), pose );
        std::cout << "OK" << std::endl;

        std::cout << "generating depth image...";
        raycaster_ptr_->generateDepthImage( depth_view_device_ );
        std::cout << "OK" << std::endl;

        int c;
        depth_view_device_.download( depth_view_host_, c );
        viewerDepth_->showShortImage (&depth_view_host_[0], depth_view_device_.cols(), depth_view_device_.rows(), 0, 5000, true);
        viewerDepth_->spinOnce();
    }

    void
    TSDFViewer::showGeneratedRayImage ( pcl::gpu::TsdfVolume::Ptr const& volume, const Eigen::Affine3f& pose )
    {
        //pose.linee
        std::cout << "Running raycaster...";
        raycaster_ptr_->run( *volume.get(), pose );
        std::cout << "OK" << std::endl;

        std::cout << "generating ray image...";
        raycaster_ptr_->generateSceneView( ray_view_device_ );
        std::cout << "OK" << std::endl;

        int c;
        std::cout << "Downloading ray image...";
        ray_view_device_.download( ray_view_host_, c );
        std::cout << "OK" << std::endl;

        std::cout << "Showing ray image...";
        viewerScene_->showRGBImage (reinterpret_cast<unsigned char*>(&ray_view_host_[0]), ray_view_device_.cols(), ray_view_device_.rows() );
        std::cout << "OK" << std::endl;

        viewerScene_->spinOnce();

    }

    void
    TSDFViewer::toCloud()
    {
        pcl::PointCloud<pcl::PointXYZI>::Ptr cloud;
        std::cout << "converting to cloud";
        tsdf_volume_.convertToTsdfCloud( cloud );
        std::cout << "OK" << std::endl;
    }

#if 0
    void
    TSDFViewer::showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, Eigen::Affine3f* pose_ptr )
    {
        if ( pose_ptr )
        {
            raycaster_ptr_->run(kinfu.volume(), *pose_ptr);
            raycaster_ptr_->generateSceneView(view_device_);
        }
        //else
        //    kinfu.getImage (view_device_);

        /*if (paint_image_ && registration && !pose_ptr)
        {
            colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
            paint3DView (colors_device_, view_device_);
        }*/


        int cols;
        view_device_.download (view_host_, cols);
        viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());

        //viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);

        /*#ifdef HAVE_OPENCV
        if (accumulate_views_)
        {
            views_.push_back (cv::Mat ());
            cv::cvtColor (cv::Mat (480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
            //cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
        }
        #endif*/
    }


#endif

} // ns am
