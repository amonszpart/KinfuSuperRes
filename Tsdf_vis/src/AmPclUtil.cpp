#include "AmPclUtil.h"

#include <iostream>

namespace am
{
    namespace util
    {
        namespace pcl
        {

            void
            setViewerPose ( ::pcl::visualization::PCLVisualizer& viewer, const Eigen::Matrix4f& p_viewer_pose )
            {
                Eigen::Affine3f viewer_pose( p_viewer_pose );
                setViewerPose( viewer, viewer_pose );
            }

            void
            setViewerPose ( ::pcl::visualization::PCLVisualizer& viewer, const Eigen::Affine3f & viewer_pose )
            {
                Eigen::Vector3f pos_vector     = viewer_pose * Eigen::Vector3f (0, 0, 0);
                Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
                Eigen::Vector3f up_vector      = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
                viewer.setCameraPosition(
                            pos_vector[0], pos_vector[1], pos_vector[2],
                            look_at_vector[0], look_at_vector[1], look_at_vector[2],
                            up_vector[0], up_vector[1], up_vector[2] );
            }

            Eigen::Vector3f
            point2To3D( Eigen::Vector2f const& pnt2, Eigen::Matrix3f const& intrinsics )
            {
                Eigen::Vector3f pnt3D;
                pnt3D << (pnt2(0) - intrinsics(0,2)) / intrinsics(0,0),
                         (pnt2(1) - intrinsics(1,2)) / intrinsics(1,1),
                         1.f;
                return pnt3D;
            }

            Eigen::Vector2f
            point3To2D( Eigen::Vector3f const& pnt3, Eigen::Matrix3f const& intrinsics )
            {
                Eigen::Vector2f pnt2;
                pnt2 << (pnt3(0) / pnt3(2)) * intrinsics(0,0) + intrinsics(0,2),
                        (pnt3(1) / pnt3(2)) * intrinsics(1,1) + intrinsics(1,2);
                return pnt2;
            }

            void printPose( Eigen::Affine3f const& pose )
            {
                // debug
                std::cout << pose.linear() << std::endl <<
                             pose.translation().transpose() << std::endl;

                float alpha = atan2(  pose.linear()(1,0), pose.linear()(0,0) );
                float beta  = atan2( -pose.linear()(2,0),
                                     sqrt( pose.linear()(2,1) * pose.linear()(2,1) +
                                           pose.linear()(2,2) * pose.linear()(2,2)  )
                                     );
                float gamma = atan2(  pose.linear()(2,1), pose.linear()(2,2) );

                std::cout << "alpha: " << alpha << " " << alpha * 180.f / M_PI << std::endl;
                std::cout << "beta: "  << beta  << " " << beta  * 180.f / M_PI << std::endl;
                std::cout << "gamma: " << gamma << " " << gamma * 180.f / M_PI << std::endl;
            }
        } // end ns pcl
    } // end ns util
} // end ns am
