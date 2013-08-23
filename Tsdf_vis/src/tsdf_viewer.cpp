#include "tsdf_viewer.h"
#include "../kinfu/tools/kinfu_app.h"
#include "../kinfu/tools/kinfu_util.h"
#include <pcl/PolygonMesh.h>
#include <vector_types.h> // cuda-5.0

#include <pcl/range_image/range_image_planar.h>
#include <pcl/visualization/range_image_visualizer.h>

#include <vtkPNGWriter.h>
#include <vtkWindow.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageShiftScale.h>
#include <vtkCamera.h>
#include <vtkImageData.h>

namespace am
{

    TSDFViewer::TSDFViewer( bool no_tsdf )
    {
        const int rows = 2*480;
        const int cols = 2*640;

        intr_ = MyIntr( 521.7401 * 2.f, 522.1379 * 2.f,
                        323.4402 * 2.f, 258.1387 * 2.f );

        //initCloudViewer( rows, cols );
        if ( !no_tsdf )
        {
            initRayViewer( rows, cols );
            initDepthViewer( rows, cols );
        }
        initCloudViewer( rows, cols );

        //range_vis_ = pcl::visualization::RangeImageVisualizer::Ptr( new pcl::visualization::RangeImageVisualizer("range image"));
        //range_vis_ = pcl::visualization::RangeImageVisualizer::getRangeImageWidget(
        //range_vis_->setSize( cols, rows );
    }

    void
    TSDFViewer::loadTsdfFromFile( std::string path, bool binary )
    {
        tsdf_volume_.load( path, binary );
        std::cout << "initing kinfuvolume with resolution: " << tsdf_volume_.gridResolution().transpose() << std::endl;
        //tsdf_volume_.header() = pc
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
        //boost::shared_ptr<pcl::PolygonMesh> mesh_ptr;
        extractMeshFromVolume( kinfuVolume_ptr_, mesh_ptr_ );

        // save mesh
        if ( mesh_ptr_.get() != NULL )
        {
            pcl::io::savePLYFile( path + "_mesh.ply", *mesh_ptr_ );
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
        //cloud_viewer_->addCoordinateSystem (1.0);
        cloud_viewer_->initCameraParameters ();
        cloud_viewer_->setPosition (0, 500);
        cloud_viewer_->setSize (cols, rows);
        cloud_viewer_->setCameraClipDistances( 0.001, 10.01 );
        cloud_viewer_->setShowFPS( false );
        //cloud_viewer_->setCameraParameters( 521.7401, 522.1379, 323.4402 * 2.f, 258.1387 *2.f);



        // Compute the vertical field of view based on the focal length and image height
        double fovy = 2.0 * atan(intr_.cy / 2.f / intr_.fy) * 180.0 / M_PI;
        std::cout << "fovy: " << fovy << std::endl;

        vtkSmartPointer<vtkRendererCollection> rens = cloud_viewer_->getRendererCollection();
        rens->InitTraversal ();
        vtkRenderer* renderer = NULL;
        int i = 1;
        while ((renderer = rens->GetNextItem ()) != NULL)
        {
            vtkSmartPointer<vtkCamera> cam = renderer->GetActiveCamera ();
            cam->SetUseHorizontalViewAngle (0);
            // Description:
            // Set/Get the camera view angle, which is the angular height of the
            // camera view measured in degrees.  The default angle is 30 degrees.
            // This method has no effect in parallel projection mode.
            // The formula for setting the angle up for perfect perspective viewing
            // is: angle = 2*atan((h/2)/d) where h is the height of the RenderWindow
            // (measured by holding a ruler up to your screen) and d is the
            // distance from your eyes to the screen.
            int *renderwindowsize = renderer->GetRenderWindow()->GetSize();
            std::cout << "renderwindow.size: " << renderwindowsize[0] << " " << renderwindowsize[1] << std::endl;
            std::cout << "pixelaspect: " << *renderer->GetPixelAspect() << std::endl;
            double *aspect = renderer->GetAspect();
            std::cout << "aspect: " << aspect[0] << aspect[1] << std::endl;
            std::cout << "fovy before: " << cam->GetViewAngle() << std::endl;

            cam->SetViewAngle(fovy);

            double real_fovy = cam->GetViewAngle();
            std::cout << "fovy after: " << real_fovy << std::endl;
            aspect = renderer->GetAspect();
            std::cout << "aspect: " << aspect[0] << "," << aspect[1] << std::endl;
            std::cout << "pixelaspect: " << renderer->GetPixelAspect()[0] << "," << renderer->GetPixelAspect()[1] << std::endl;
            std::cout << "renderwindow.size: " << renderer->GetRenderWindow()->GetSize()[0] <<
                         " " << renderer->GetRenderWindow()->GetSize()[1] << std::endl;


            // THIS IS SETTING THE PROJ MATRIX USING the already set VIEWANGLE
            vtkMatrix4x4 *projmat = cam->GetProjectionTransformMatrix( /* aspect (w/h): */ intr_.cx/intr_.cy, 0.001, 10.01 );

            //vtkMatrix4x4 *projmat = cam->GetProjectionTransformMatrix( 4./3., 0.01, 10.01 );
            //cam->SetParallelProjection();
            std::cout << "projmat: " << *projmat << std::endl;
            real_fovy = cam->GetViewAngle();
            std::cout << "fovy after: " << real_fovy << std::endl;
            aspect = renderer->GetAspect();
            std::cout << "aspect: " << aspect[0] << "," << aspect[1] << std::endl;
            std::cout << "pixelaspect: " << renderer->GetPixelAspect()[0] << "," << renderer->GetPixelAspect()[1] << std::endl;
            std::cout << "renderwindow.size: " << renderer->GetRenderWindow()->GetSize()[0] <<
                         " " << renderer->GetRenderWindow()->GetSize()[1] << std::endl;

        }

        //cloud_viewer_->addText ("H: print help", 2, 15, 20, 34, 135, 246);
    }

    void
    TSDFViewer::setViewerFovy( pcl::visualization::PCLVisualizer &viewer, Eigen::Matrix3f const& intr )
    {
        // Compute the vertical field of view based on the focal length and image height
        double im_height = intr(1,2) / 2.f;
        double fovy = 2.0 * atan( im_height / intr(1,1) ) * 180.0 / M_PI;
        std::cout << "fovy: " << fovy << std::endl;

        vtkSmartPointer<vtkRendererCollection> rens = viewer.getRendererCollection();
        rens->InitTraversal ();
        vtkRenderer* renderer = NULL;
        int i = 1;
        while ( (renderer = rens->GetNextItem()) != NULL )
        {
            vtkSmartPointer<vtkCamera> cam = renderer->GetActiveCamera ();
            cam->SetUseHorizontalViewAngle (0);
            cam->SetViewAngle(fovy);
        }
    }

    void
    TSDFViewer::initRayCaster( int rows, int cols )
    {
        raycaster_ptr_ = RayCaster::Ptr( new RayCaster( rows, cols ) );
        raycaster_ptr_->setIntrinsics(  521.7401 * 2.f, 522.1379 * 2.f, 323.4402 * 2.f, 258.1387 *2.f );
    }

    void
    TSDFViewer::initRayViewer( int rows, int cols )
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 0);
        if ( !raycaster_ptr_ ) initRayCaster( rows, cols );
    }

    void
    TSDFViewer::initDepthViewer( int rows, int cols )
    {
        viewerDepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerDepth_->setWindowTitle ("Depth from ray tracing");
        viewerDepth_->setPosition (640, 0);


        if ( !raycaster_ptr_ ) initRayCaster( rows, cols );
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
    TSDFViewer::toCloud( Eigen::Affine3f const& pose, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_ptr )
    {
        if ( !cloud_ptr.get() )
            cloud_ptr = pcl::PointCloud<pcl::PointXYZI>::Ptr( new pcl::PointCloud<pcl::PointXYZI>() );

        std::cout << "converting to cloud";
        tsdf_volume_.convertToTsdfCloud( cloud_ptr );
        std::cout << "OK" << std::endl;
    }

    void
    TSDFViewer::showCloud( Eigen::Affine3f const& pose, pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud_ptr )
    {
        cloud_viewer_->removeAllPointClouds();
        //cloud_viewer_->setCameraPosition();
        //pcl::visualization::PointCloudColorHandler<PointXYZI> cld( cloud_ptr_ );
        cloud_viewer_->addPointCloud<pcl::PointXYZI>( cloud_ptr, std::string("cloud") );
        cloud_viewer_->spinOnce(1);
    }

    void
    TSDFViewer::showMesh( Eigen::Affine3f const& pose, pcl::PolygonMesh::Ptr & mesh_ptr )
    {
        cloud_viewer_->removeAllPointClouds();
        cloud_viewer_->addPolygonMesh( *mesh_ptr );
        cloud_viewer_->spinOnce(1);
    }

    void
    TSDFViewer::renderRangeImage( pcl::PointCloud<pcl::PointXYZI>::Ptr const& cloud, Eigen::Affine3f const& pose )
    {
        float angularResolution = (float) (  .05f * (M_PI/180.0f));  //   1.0 degree in radians
        float maxAngleWidth     = (float) (180.0f * (M_PI/180.0f));  // 360.0 degree in radians
        float maxAngleHeight    = (float) (180.0f * (M_PI/180.0f));  // 180.0 degree in radians
        //Eigen::Affine3f sensorPose = (Eigen::Affine3f)Eigen::Translation3f(0.0f, 0.0f, -3.0f);
        pcl::RangeImage::CoordinateFrame coordinate_frame = pcl::RangeImage::CAMERA_FRAME;
        float noiseLevel = 0.00;
        float minRange = 0.0f;
        int borderSize = 0;

        pcl::RangeImage rangeImage;
        rangeImage.createFromPointCloud(*cloud, angularResolution, maxAngleWidth, maxAngleHeight,
                                        pose, coordinate_frame, noiseLevel, minRange, borderSize);

        std::cout << rangeImage << "\n";

        saveRangeImagePlanarFilePNG( std::string("rangeImage.png"), rangeImage );
        //        ((pcl::visualization::RangeImageVisualizer*)range_vis_.get())->showRangeImage( rangeImage );
        if ( range_vis_ )
            range_vis_.reset();
        range_vis_ = pcl::visualization::RangeImageVisualizer::Ptr(
                         pcl::visualization::RangeImageVisualizer::getRangeImageWidget( rangeImage,0.f, 5000.f, true ) );
    }

    // http://www.pcl-users.org/Writing-a-pcl-RangeImage-to-an-image-png-file-td3724081.html
    void
    TSDFViewer::saveRangeImagePlanarFilePNG ( std::string const& file_name, pcl::RangeImage const& range_image )
    {
        std::cout << "saving range image to " << file_name << "...";
        cv::Mat mat( range_image.height, range_image.width, CV_32FC1 );
        for (int y = 0; y < range_image.height; ++y)
        {
            for (int x = 0; x < range_image.width; ++x)
            {
                mat.at<float>(y,x) = range_image( /*col: */ x, /* row: */ y).range / 5000.f * 255.f;
                //std::cout << mat.at<float>(y,x) << std::endl;
            }
        }
        std::vector<int> params;
        params.push_back( 16 );
        params.push_back( 0 );
        cv::imwrite( file_name.c_str(), mat, params );
        std::cout << "OK" << std::endl;
    }

    void
    TSDFViewer::setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
    {
        Eigen::Vector3f pos_vector     = viewer_pose * Eigen::Vector3f (0, 0, 0);
        Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
        Eigen::Vector3f up_vector      = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
        viewer.setCameraPosition(
                    pos_vector[0], pos_vector[1], pos_vector[2],
                    look_at_vector[0], look_at_vector[1], look_at_vector[2],
                    up_vector[0], up_vector[1], up_vector[2] );
    }

    Eigen::Affine3f
    TSDFViewer::getViewerPose ( visualization::PCLVisualizer& viewer )
    {
        Eigen::Affine3f pose = viewer.getViewerPose();
        Eigen::Matrix3f rotation = pose.linear();

        Matrix3f axis_reorder;
        axis_reorder << 0,  0,  1,
                -1,  0,  0,
                0, -1,  0;

        rotation = rotation * axis_reorder;
        pose.linear() = rotation;
        return pose;
    }

    Eigen::Vector3f
    TSDFViewer::getViewerCameraUp( pcl::visualization::PCLVisualizer& viewer )
    {
        std::vector<pcl::visualization::Camera> cameras;
        viewer.getCameras( cameras );
        if ( cameras.size() > 0 )
        {
            return Eigen::Vector3f( cameras[0].view[0], cameras[0].view[1], cameras[0].view[2] );
        }
        return Eigen::Vector3f::Zero();
    }

    //#include <vtk/
    void
    TSDFViewer::fetchVtkZBuffer( std::vector<float> &data, int &w, int &h )
    {
        std::cout << "saving vtkZBuffer...";
        vtkSmartPointer<vtkWindowToImageFilter> filter =
                vtkSmartPointer<vtkWindowToImageFilter>::New();
        vtkSmartPointer<vtkPNGWriter> imageWriter =
                vtkSmartPointer<vtkPNGWriter>::New();
        vtkSmartPointer<vtkImageShiftScale> scale =
                vtkSmartPointer<vtkImageShiftScale>::New();

        vtkSmartPointer<vtkRenderWindow> renWin = cloud_viewer_->getRenderWindow();

        // Create Depth Map
        filter->SetInput( renWin );
        filter->SetMagnification(1);
        filter->SetInputBufferTypeToZBuffer();        //Extract z buffer value

        scale->SetOutputScalarTypeToFloat();
        scale->SetInputConnection(filter->GetOutputPort());
        scale->SetShift(0.f);
        //scale->SetScale(-65535.f);
        scale->SetScale(1.f);
        scale->Update();

        vtkSmartPointer<vtkImageData> imageData = scale->GetOutput();
        int* dims = imageData->GetDimensions();
        // int dims[3]; // can't do this

        std::cout << "Dims: " << " x: " << dims[0] << " y: " << dims[1] << " z: " << dims[2] << std::endl;

        std::cout << "Number of points: " << imageData->GetNumberOfPoints() << std::endl;
        std::cout << "Number of cells: " << imageData->GetNumberOfCells() << std::endl;

        data.resize( dims[0] * dims[1] * dims[2] );
        // Retrieve the entries from the image data and print them to the screen
        for (int z = 0; z < dims[2]; z++)
        {
            for (int y = 0; y < dims[1]; ++y)
            {
                for (int x = 0; x < dims[0]; x++)
                {
                    float* pixel = static_cast<float*>( imageData->GetScalarPointer(x,y,z) );
                    // do something with v
                    //std::cout << pixel[0] << " ";
                    data[ z * dims[1] * dims[0] + (dims[1] - y - 1) * dims[0] + x ] = pixel[0];//(pixel[0] == 10001) ? 0 : pixel[0];
                }
                //std::cout << std::endl;
            }
            //std::cout << std::endl;
        }
        w = dims[0];
        h = dims[1];

        // Write depth map as a .bmp image
        imageWriter->SetFileName( "vtkzbuffer.png" );
        imageWriter->SetInputConnection(scale->GetOutputPort());
        imageWriter->Write();
        std::cout << "OK" << std::endl;

        //if ( dims ) delete [] dims;
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
