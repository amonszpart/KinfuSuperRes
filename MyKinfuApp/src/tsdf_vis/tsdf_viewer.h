#ifndef TSDF_VIEWER_H
#define TSDF_VIEWER_H

#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/visualization/pcl_visualizer.h>

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
            pcl::gpu::KinfuTracker kinfu_;
            pcl::TSDFVolume<float, short> tsdf_volume_;

            void
            loadTsdfFromFile( std::string path, bool binary );
            void
            extractMeshFromVolume( const pcl::gpu::TsdfVolume::Ptr volume, boost::shared_ptr<pcl::PolygonMesh> mesh_ptr );

        protected:
            pcl::gpu::MarchingCubes::Ptr marching_cubes_;
            pcl::visualization::PCLVisualizer::Ptr cloud_viewer_;

            void
            initCloudViewer();
    };

} // ns am

#endif // TSDF_VIEWER_H
