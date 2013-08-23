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
            run( /* out: */ cv::Mat &depth,
                 /*  in: */ pcl::PolygonMesh::Ptr &meshPtr, Eigen::Affine3f const& pose, int subdivIterations = 2 );

            void
            enhanceMesh( /* out: */ pcl::PolygonMesh::Ptr &outMeshPtr,
                         /*  in: */ cv::Mat const& dep16, pcl::PolygonMesh::ConstPtr const& inMeshPtr, Eigen::Affine3f const& p_pose,
                         const float resolution = 3.f / 512.f );

            void
            rayCast( /* out: */ std::vector<int> &p_indices, cv::Mat *p_depth,
                     /*  in: */ Octree::Ptr const& octree, pcl::PointCloud<PointT>::Ptr const& cloudPtr, Eigen::Affine3f const& pose );

            void
            calculatePointDiffs( /* out: */
                                 /*  in: */
                                 OctreePtr const& octreePtr,
                                 pcl::PointCloud<PointT>::Ptr const& cloudPtr,
                                 Eigen::Affine3f const& pose,
                                 cv::Mat const& depth );

            static void
            subdivideMesh( /* out: */ pcl::PolygonMesh &output_mesh,
                           /*  in: */ pcl::PolygonMesh::ConstPtr input_mesh, int iterations = 1 );

            static void
            mesh2Octree( /* out: */ Octree::Ptr &octreePtr, pcl::PointCloud<PointT>::Ptr &cloudPtr,
                         /*  in: */ pcl::PolygonMesh::ConstPtr mesh, float resolution = .05f );

        protected:
            Octree::Ptr octree_ptr_;
            pcl::PointCloud<PointT>::Ptr cloud_ptr_;

    };

} // end ns am

#endif // MESHRAYCASTER_H
