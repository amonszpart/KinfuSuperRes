#ifndef AMPCLUTIL_H
#define AMPCLUTIL_H

#include <eigen3/Eigen/Dense>
#include <pcl/visualization/pcl_visualizer.h>

#include <opencv2/core/core.hpp>

namespace am
{
    namespace util
    {
        namespace pcl
        {
            void
            fetchViewerZBuffer( /* out: */ cv::Mat & zBufMat,
                                /*  in: */ ::pcl::visualization::PCLVisualizer::Ptr const& viewer, double zNear = 0.001, double zFar = 10.01 );

            void
            setViewerPose ( ::pcl::visualization::PCLVisualizer& viewer, Eigen::Matrix4f const& p_viewer_pose );
            void
            setViewerPose ( ::pcl::visualization::PCLVisualizer& viewer, Eigen::Affine3f const& viewer_pose );

            Eigen::Vector3f
            point2To3D( Eigen::Vector2f const& pnt, Eigen::Matrix3f const& intrinsics );
            Eigen::Vector2f
            point3To2D( Eigen::Vector3f const& pnt, Eigen::Matrix3f const& intrinsics );

            void
            printPose( Eigen::Affine3f const& pose );

            void
            setCam( /*  in: */ Eigen::Vector3d &pos, Eigen::Vector3d &up, Eigen::Vector3d &dir,
                    ::pcl::visualization::PCLVisualizer::Ptr pViewerPtr );
            void
            getCam( /* out: */ Eigen::Vector3d &pos, Eigen::Vector3d &up, Eigen::Vector3d &dir,
                    /*  in: */ ::pcl::visualization::PCLVisualizer::Ptr  const& pViewerPtr );

            void copyCam( ::pcl::visualization::PCLVisualizer::Ptr from,
                          ::pcl::visualization::PCLVisualizer::Ptr to );

        } // pcl

        namespace os
        {
            void get_by_extension_in_dir( std::vector<boost::filesystem::path>& ret,
                                          boost::filesystem::path const& root,
                                          std::string const& ext,
                                          std::string const* beginsWith = NULL );
        } // end ns os
    } // util
} // am


#endif // AMPCLUTIL_H
