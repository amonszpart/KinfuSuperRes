#ifndef TSDF_VIEWER_H
#define TSDF_VIEWER_H

//#include <pcl/gpu/kinfu/kinfu.h>
#include "kinfu.h"
#include "marching_cubes.h"
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/range_image/range_image.h>

#include "../kinfu/tools/tsdf_volume.h"
#include "../kinfu/tools/tsdf_volume.hpp"

namespace pcl
{
    struct PolygonMesh;
}

namespace am
{
    class TSDFViewer
    {
        public:
            TSDFViewer();
            //pcl::gpu::KinfuTracker kinfu_;
            pcl::TSDFVolume<float, short> tsdf_volume_; // tmp read in storage
            pcl::gpu::TsdfVolume::Ptr kinfuVolume_ptr_;

            void
            loadTsdfFromFile( std::string path, bool binary );
            void initRayCaster( int rows, int cols );

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
            toCloud(const Eigen::Affine3f &pose);
            void
            saveRangeImagePlanarFilePNG( const std::string &file_name,  pcl::RangeImage const& range_image );

            std::vector<unsigned short> const& getLatestDepth() const { return depth_view_host_; };

            std::vector<pcl::gpu::KinfuTracker::PixelRGB> const& getLatestRayCast() const { return ray_view_host_; }

            pcl::visualization::ImageViewer::Ptr & getRayViewer() { return viewerScene_; }
            pcl::visualization::ImageViewer::Ptr & getDepthViewer() { return viewerDepth_; }

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

            void
            initCloudViewer(int rows, int cols);
            void
            initRayViewer( int rows, int cols );
            void
            initDepthViewer( int rows, int cols );

    };

} // ns am

#endif // TSDF_VIEWER_H
