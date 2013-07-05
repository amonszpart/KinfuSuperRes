#ifndef __SCENE_CLOUD_VIEW_H
#define __SCENE_CLOUD_VIEW_H

#include "kinfu_pcl_headers.h"
#include "point_cloud_color_handler_rgb_cloud.hpp"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace pcl
{
    namespace gpu
    {
        void mergePointNormal (const DeviceArray<PointXYZ>& cloud, const DeviceArray<Normal>& normals, DeviceArray<PointNormal>& output);
    }
}

namespace am
{

    // CLASS
    struct SceneCloudView
    {
            enum { GPU_Connected6 = 0, CPU_Connected6 = 1, CPU_Connected26 = 2 };

            SceneCloudView(int viz);

            void
            show (KinfuTracker& kinfu, bool integrate_colors);

            void
            toggleCube(const Eigen::Vector3f& size);

            void
            toggleExtractionMode ();

            void
            toggleNormals ();

            void
            clearClouds (bool print_message = false);

            void
            showMesh(KinfuTracker& kinfu, bool /*integrate_colors*/);

            int viz_;
            int extraction_mode_;
            bool compute_normals_;
            bool valid_combined_;
            bool cube_added_;

            Eigen::Affine3f viewer_pose_;

            visualization::PCLVisualizer::Ptr cloud_viewer_;

            PointCloud<PointXYZ>::Ptr cloud_ptr_;
            PointCloud<Normal>::Ptr normals_ptr_;

            DeviceArray<PointXYZ> cloud_buffer_device_;
            DeviceArray<Normal> normals_device_;

            PointCloud<PointNormal>::Ptr combined_ptr_;
            DeviceArray<PointNormal> combined_device_;

            DeviceArray<RGB> point_colors_device_;
            PointCloud<RGB>::Ptr point_colors_ptr_;

            MarchingCubes::Ptr marching_cubes_;
            DeviceArray<PointXYZ> triangles_buffer_device_;

            boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
    };

} // ns am

#endif
