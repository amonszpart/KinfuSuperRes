#ifndef AMPCLUTIL_H
#define AMPCLUTIL_H

#include <eigen3/Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/PolygonMesh.h>

#include <opencv2/core/core.hpp>
#include <boost/filesystem.hpp>

namespace am
{
    namespace util
    {
        namespace pcl
        {
            extern void
            fetchViewerZBuffer( /* out: */ cv::Mat & zBufMat,
                                /*  in: */ ::pcl::visualization::PCLVisualizer::Ptr const& viewer, double zNear = 0.001, double zFar = 10.01 );

            extern void
            setViewerPose ( ::pcl::visualization::PCLVisualizer& viewer, Eigen::Matrix4f const& p_viewer_pose );
            extern void
            setViewerPose ( ::pcl::visualization::PCLVisualizer& viewer, Eigen::Affine3f const& viewer_pose );

            extern Eigen::Vector3f
            point2To3D( Eigen::Vector2f const& pnt, Eigen::Matrix3f const& intrinsics );
            extern Eigen::Vector2f
            point3To2D( Eigen::Vector3f const& pnt, Eigen::Matrix3f const& intrinsics );

            extern void
            printPose( Eigen::Affine3f const& pose );

            extern void
            setCam( /*  in: */ Eigen::Vector3d &pos, Eigen::Vector3d &up, Eigen::Vector3d &dir,
                    ::pcl::visualization::PCLVisualizer::Ptr pViewerPtr );
            extern void
            getCam( /* out: */ Eigen::Vector3d &pos, Eigen::Vector3d &up, Eigen::Vector3d &dir,
                    /*  in: */ ::pcl::visualization::PCLVisualizer::Ptr  const& pViewerPtr );

            extern void
            copyCam( ::pcl::visualization::PCLVisualizer::Ptr from,
                     ::pcl::visualization::PCLVisualizer::Ptr to );

            extern void
            addFace( ::pcl::PolygonMesh::Ptr &meshPtr, std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f> *colors );

        } // pcl

        namespace os
        {
            extern void
            get_by_extension_in_dir( std::vector<boost::filesystem::path>& ret,
                                     boost::filesystem::path const& root,
                                     std::string const& ext,
                                     std::string const* beginsWith = NULL );
        } // end ns os
    } // util
} // am


#endif // AMPCLUTIL_H
