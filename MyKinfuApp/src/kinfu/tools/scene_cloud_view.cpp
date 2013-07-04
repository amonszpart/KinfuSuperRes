#include "scene_cloud_view.h"

namespace am
{

    boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
    {
        if (triangles.empty())
            return boost::shared_ptr<pcl::PolygonMesh>();

        pcl::PointCloud<pcl::PointXYZ> cloud;
        cloud.width  = (int)triangles.size();
        cloud.height = 1;
        triangles.download(cloud.points);

        boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() );
        pcl::toROSMsg(cloud, mesh_ptr->cloud);

        mesh_ptr->polygons.resize (triangles.size() / 3);
        for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
        {
            pcl::Vertices v;
            v.vertices.push_back(i*3+0);
            v.vertices.push_back(i*3+2);
            v.vertices.push_back(i*3+1);
            mesh_ptr->polygons[i] = v;
        }
        return mesh_ptr;
    }

    SceneCloudView::SceneCloudView(int viz)
        : viz_(viz), extraction_mode_ (GPU_Connected6), compute_normals_ (false), valid_combined_ (false), cube_added_(false)
    {
        cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);
        normals_ptr_ = PointCloud<Normal>::Ptr (new PointCloud<Normal>);
        combined_ptr_ = PointCloud<PointNormal>::Ptr (new PointCloud<PointNormal>);
        point_colors_ptr_ = PointCloud<RGB>::Ptr (new PointCloud<RGB>);

        if (viz_)
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
    }

    void
    SceneCloudView::show (KinfuTracker& kinfu, bool integrate_colors)
    {
        viewer_pose_ = kinfu.getCameraPose();

        ScopeTimeT time ("PointCloud Extraction");
        cout << "\nGetting cloud... " << flush;

        valid_combined_ = false;

        if (extraction_mode_ != GPU_Connected6)     // So use CPU
        {
            kinfu.volume().fetchCloudHost (*cloud_ptr_, extraction_mode_ == CPU_Connected26);
        }
        else
        {
            DeviceArray<PointXYZ> extracted = kinfu.volume().fetchCloud (cloud_buffer_device_);

            if (compute_normals_)
            {
                kinfu.volume().fetchNormals (extracted, normals_device_);
                pcl::gpu::mergePointNormal (extracted, normals_device_, combined_device_);
                combined_device_.download (combined_ptr_->points);
                combined_ptr_->width = (int)combined_ptr_->points.size ();
                combined_ptr_->height = 1;

                valid_combined_ = true;
            }
            else
            {
                extracted.download (cloud_ptr_->points);
                cloud_ptr_->width = (int)cloud_ptr_->points.size ();
                cloud_ptr_->height = 1;
            }

            if (integrate_colors)
            {
                kinfu.colorVolume().fetchColors(extracted, point_colors_device_);
                point_colors_device_.download(point_colors_ptr_->points);
                point_colors_ptr_->width = (int)point_colors_ptr_->points.size ();
                point_colors_ptr_->height = 1;
            }
            else
                point_colors_ptr_->points.clear();
        }
        size_t points_size = valid_combined_ ? combined_ptr_->points.size () : cloud_ptr_->points.size ();
        cout << "Done.  Cloud size: " << points_size / 1000 << "K" << endl;

        if (viz_)
        {
            cloud_viewer_->removeAllPointClouds ();
            if (valid_combined_)
            {
                visualization::PointCloudColorHandlerRGBCloud<PointNormal> rgb(combined_ptr_, point_colors_ptr_);
                cloud_viewer_->addPointCloud<PointNormal> (combined_ptr_, rgb, "Cloud");
                cloud_viewer_->addPointCloudNormals<PointNormal>(combined_ptr_, 50);
            }
            else
            {
                visualization::PointCloudColorHandlerRGBCloud<PointXYZ> rgb(cloud_ptr_, point_colors_ptr_);
                cloud_viewer_->addPointCloud<PointXYZ> (cloud_ptr_, rgb);
            }
        }
    }

    void
    SceneCloudView::toggleCube(const Eigen::Vector3f& size)
    {
        if (!viz_)
            return;

        if (cube_added_)
            cloud_viewer_->removeShape("cube");
        else
            cloud_viewer_->addCube(size*0.5, Eigen::Quaternionf::Identity(), size(0), size(1), size(2));

        cube_added_ = !cube_added_;
    }

    void
    SceneCloudView::toggleExtractionMode ()
    {
        extraction_mode_ = (extraction_mode_ + 1) % 3;

        switch (extraction_mode_)
        {
            case 0: cout << "Cloud exctraction mode: GPU, Connected-6" << endl; break;
            case 1: cout << "Cloud exctraction mode: CPU, Connected-6    (requires a lot of memory)" << endl; break;
            case 2: cout << "Cloud exctraction mode: CPU, Connected-26   (requires a lot of memory)" << endl; break;
        }
        ;
    }

    void
    SceneCloudView::toggleNormals ()
    {
        compute_normals_ = !compute_normals_;
        cout << "Compute normals: " << (compute_normals_ ? "On" : "Off") << endl;
    }

    void
    SceneCloudView::clearClouds (bool print_message)
    {
        if (!viz_)
            return;

        cloud_viewer_->removeAllPointClouds ();
        cloud_ptr_->points.clear ();
        normals_ptr_->points.clear ();
        if (print_message)
            cout << "Clouds/Meshes were cleared" << endl;
    }

    void
    SceneCloudView::showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/)
    {
        if (!viz_)
            return;

        ScopeTimeT time ("Mesh Extraction");
        cout << "\nGetting mesh... " << flush;

        if (!marching_cubes_)
            marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

        DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu.volume(), triangles_buffer_device_);
        mesh_ptr_ = convertToMesh(triangles_device);

        cloud_viewer_->removeAllPointClouds ();
        if (mesh_ptr_)
            cloud_viewer_->addPolygonMesh(*mesh_ptr_);

        cout << "Done.  Triangles number: " << triangles_device.size() / MarchingCubes::POINTS_PER_TRIANGLE / 1000 << "K" << endl;
    }

} // ns am
