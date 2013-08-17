#ifndef TSDF_VIEWER_H
#define TSDF_VIEWER_H

//#include <pcl/gpu/kinfu/kinfu.h>
#include "kinfu.h"
#include "marching_cubes.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl/range_image/range_image.h>

//#include "internal.h"
#include "../kinfu/tools/tsdf_volume.h"
#include "../kinfu/tools/tsdf_volume.hpp"

namespace pcl
{
    struct PolygonMesh;
}

namespace am
{
    struct MyIntr
    {
      float fx, fy, cx, cy;
      MyIntr () {}
      MyIntr (float fx_, float fy_, float cx_, float cy_) : fx(fx_), fy(fy_), cx(cx_), cy(cy_) {}

      MyIntr operator()(int level_index) const
      {
        int div = 1 << level_index;
        return (MyIntr (fx / div, fy / div, cx / div, cy / div));
      }
    };

    class TSDFViewer
    {
        public:
            TSDFViewer();
            //pcl::gpu::KinfuTracker kinfu_;
            pcl::TSDFVolume<float, short> tsdf_volume_; // tmp read in storage
            pcl::gpu::TsdfVolume::Ptr kinfuVolume_ptr_;

            am::MyIntr intr_;

            void
            loadTsdfFromFile( std::string path, bool binary );
            void
            initRayCaster( int rows, int cols );

            void
            showGeneratedDepth (const pcl::gpu::TsdfVolume::Ptr &volume, const Eigen::Affine3f& pose );
            void
            showGeneratedRayImage ( pcl::gpu::TsdfVolume::Ptr const& volume, const Eigen::Affine3f& pose );
            void
            spin() { viewerScene_->spin(); };
            void
            spinOnce( int time = 1) { viewerScene_->spinOnce(time); };

            // mesh
            void
            extractMeshFromVolume( const pcl::gpu::TsdfVolume::Ptr volume, boost::shared_ptr<pcl::PolygonMesh>& mesh_ptr );
            void dumpMesh( std::string path = "" );

            // cloude
            void
            toCloud( Eigen::Affine3f const& pose, pcl::PointCloud<pcl::PointXYZI>::Ptr & cloud_ptr );
            void
            showCloud( Eigen::Affine3f const& pose, pcl::PointCloud<pcl::PointXYZI>::ConstPtr cloud_ptr );
            void
            showMesh( Eigen::Affine3f const& pose, pcl::PolygonMesh::Ptr & mesh_ptr );
            void
            renderRangeImage( pcl::PointCloud<pcl::PointXYZI>::Ptr const& cloud, Eigen::Affine3f const& pose );
            void
            saveRangeImagePlanarFilePNG( const std::string &file_name,  pcl::RangeImage const& range_image );

            std::vector<unsigned short> const& getLatestDepth() const { return depth_view_host_; };

            std::vector<pcl::gpu::KinfuTracker::PixelRGB> const& getLatestRayCast() const { return ray_view_host_; }

            pcl::visualization::ImageViewer::Ptr & getRayViewer() { return viewerScene_; }
            pcl::visualization::ImageViewer::Ptr & getDepthViewer() { return viewerDepth_; }
            pcl::visualization::PCLVisualizer::Ptr & getCloudViewer() { return cloud_viewer_; }

            pcl::PointCloud<pcl::PointXYZI>::Ptr      & CloudPtr()       { return cloud_ptr_; }
            pcl::PointCloud<pcl::PointXYZI>::Ptr const& CloudPtr() const { return cloud_ptr_; }

            pcl::PolygonMesh::Ptr      & MeshPtr()       { return mesh_ptr_; }
            pcl::PolygonMesh::Ptr const& MeshPtr() const { return mesh_ptr_; }

            static void
            setViewerPose (pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose);
            static Eigen::Affine3f
            getViewerPose (pcl::visualization::PCLVisualizer& viewer);

            void
            vtkMagic();
        protected:


            pcl::gpu::MarchingCubes::Ptr            marching_cubes_;
            pcl::visualization::PCLVisualizer::Ptr  cloud_viewer_;
            pcl::gpu::RayCaster::Ptr                raycaster_ptr_;

            pcl::gpu::KinfuTracker::View            ray_view_device_;
            std::vector<pcl::gpu::KinfuTracker::PixelRGB>          ray_view_host_;
            pcl::visualization::ImageViewer::Ptr    viewerScene_;

            pcl::gpu::KinfuTracker::DepthMap        depth_view_device_;
            std::vector<unsigned short>             depth_view_host_;
            pcl::visualization::ImageViewer::Ptr    viewerDepth_;

            pcl::PointCloud<pcl::PointXYZI>::Ptr    cloud_ptr_;
            pcl::PolygonMesh::Ptr                   mesh_ptr_;

            pcl::visualization::RangeImageVisualizer::Ptr range_vis_;

            void
            initCloudViewer(int rows, int cols);
            void
            initRayViewer( int rows, int cols );
            void
            initDepthViewer( int rows, int cols );

    };

} // ns am

#endif // TSDF_VIEWER_H
