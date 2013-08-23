#ifndef POLYMESHVIEWER_H
#define POLYMESHVIEWER_H

#include "ProcessorWithIntrinsics.h"

#include <pcl/PolygonMesh.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <eigen3/Eigen/Dense>



namespace am
{
    class PolyMeshViewer : public am::ProcessorWithIntrinsics
    {
        public:
            // init
            PolyMeshViewer( int cols = 640, int rows = 480 );
            PolyMeshViewer( Eigen::Matrix3f p_intrinsics, int cols = 640, int rows = 480 );

            void
            initViewer(std::string title = std::string("PolyMeshViewer"), int cols = -1, int rows = -1 );

            // logic
            void
            showMesh( std::string const& path, Eigen::Affine3f const& pose );
            void
            showMesh( pcl::PolygonMesh::Ptr const& meshPtr, Eigen::Affine3f const& pose );

            // events
            void
            updateToPose( Eigen::Affine3f const& pose );

            // setget
            pcl::visualization::PCLVisualizer::Ptr      & VisualizerPtr();
            pcl::visualization::PCLVisualizer::Ptr const& VisualizerPtr() const;
            pcl::PolygonMesh::Ptr      & MeshPtr();
            pcl::PolygonMesh::Ptr const& MeshPtr() const;
        protected:
            pcl::visualization::PCLVisualizer::Ptr visualizer_;
            pcl::PolygonMesh::Ptr meshPtr_;
            int rows_, cols_;
            bool inited_;
    };

} // end ns am

#endif // POLYMESHVIEWER_H

