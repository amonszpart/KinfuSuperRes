#ifndef MESHRAYCASTER_H
#define MESHRAYCASTER_H

#include "ProcessorWithIntrinsics.h"

#include <pcl/PolygonMesh.h>
#include <pcl/octree/octree.h>

#include <eigen3/Eigen/Dense>

#include <opencv2/core/core.hpp>

namespace am
{
    class MeshRayCaster : public ProcessorWithIntrinsics
    {
        public:
            typedef pcl::PointXYZ PointT;
            typedef pcl::octree::OctreePointCloudSearch<PointT> Octree;
            typedef typename Octree::Ptr OctreePtr;
            typedef typename Octree::ConstPtr OctreeConstPtr;

            MeshRayCaster()
                : ProcessorWithIntrinsics() {}
            MeshRayCaster( Eigen::Matrix3f p_intrinsics )
                : ProcessorWithIntrinsics( p_intrinsics ) {}

            void
            run( pcl::PolygonMesh::Ptr &meshPtr, Eigen::Affine3f const& pose, cv::Mat &depth );

            void
            rayCast(OctreePtr const& octree, pcl::PointCloud<PointT>::Ptr const& cloudPtr, Eigen::Affine3f const& pose,
                     std::vector<int> &p_indices, cv::Mat *p_depth );

            static void
            subdivideMesh( pcl::PolygonMesh::ConstPtr input_mesh, pcl::PolygonMesh &output_mesh, int iterations = 1 );

            static void
            mesh2Octree( pcl::PolygonMesh &mesh, Octree::Ptr &octreePtr, pcl::PointCloud<PointT>::Ptr &cloudPtr, float resolution = .05f );

        protected:
            OctreePtr octree_ptr_;
            pcl::PointCloud<PointT>::Ptr cloud_ptr_;

    };

} // end ns am

#endif // MESHRAYCASTER_H
