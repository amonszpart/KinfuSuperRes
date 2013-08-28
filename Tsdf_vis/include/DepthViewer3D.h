#ifndef DEPTHVIEWER3D_H
#define DEPTHVIEWER3D_H

#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/core.hpp>
#include "AmPclUtil.h"

#include <map>

namespace am
{

    class DepthViewer3D
    {
        public:
            // CNSTR
            DepthViewer3D();

            // METHODS
            void
            showMats( cv::Mat const& large_dep16, cv::Mat const& rgb8_960,
                      int img_id,
                      std::map<int,Eigen::Affine3f> const& poses,
                      Eigen::Matrix3f const& intrinsics  );

            // STATICS
            template <typename T>
            static void matsTo3D( cv::Mat const& dep, cv::Mat const& img,
                                  pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloudPtr,
                                  Eigen::Matrix3f const& intrinsics,
                                  float alpha, bool useColour,
                                  Eigen::Affine3f const* const pose ); /* display scaling of pointcloud */
            static void
            showAllPoses();

        protected:
            // FIELDS
            pcl::visualization::PCLVisualizer::Ptr viewer_ptr_;

    };

    template <typename depT>
    void
    DepthViewer3D::matsTo3D( cv::Mat const& dep, cv::Mat const& img,
                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloudPtr,
                             Eigen::Matrix3f const& intrinsics,
                             /* depth scale: */ float alpha, bool useColour,
                             Eigen::Affine3f const* const pose )
    {
        // check input
        if ( dep.size() != img.size() )
        {
            std::cerr << "matsTo3D(): dep and rgb need the same size! "
                      << dep.rows << "x" << dep.cols << ", "
                      << img.rows << "x" << img.cols << std::endl;
            return;
        }
        if ( img.channels() != 3 )
        {
            std::cerr << "matsTo3D(): rgb should be 3 channels! " << img.channels() << std::endl;
            return;
        }

        // allocate ouptut
        cloudPtr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr( new pcl::PointCloud<pcl::PointXYZRGB> );

        Eigen::Vector3f pnt3D;
        // copy inputs
        for ( int y = 0; y < dep.rows; ++y )
        {
            for ( int x = 0; x < dep.cols; ++x )
            {
                pnt3D = am::util::pcl::point2To3D( (Eigen::Vector2f){x,y},
                                                   intrinsics            )
                        * ( (float)dep.at<depT>( y,x ) * alpha );
                if ( pose )
                {
                    pnt3D = pose->rotation() * pnt3D + pose->translation();
                }

                pcl::PointXYZRGB point;
                point.x = pnt3D(0); //point.x = (x - intrinsics(0,2)) / intrinsics(0,0) * dep.at<depT>( y,x ) * alpha;
                point.y = pnt3D(1); //point.y = (y - intrinsics(1,2)) / intrinsics(1,1) * dep.at<depT>( y,x ) * alpha;
                point.z = pnt3D(2); //point.z = dep.at<depT>( y,x ) * alpha;

                uint32_t rgb = (static_cast<uint32_t>(img.at<uchar>(y, x * img.channels() + 2)) << 16 |
                                static_cast<uint32_t>(img.at<uchar>(y, x * img.channels() + 1)) << 8  |
                                static_cast<uint32_t>(img.at<uchar>(y, x * img.channels()    ))        );
                point.rgb = *reinterpret_cast<float*>( &rgb );
                cloudPtr->points.push_back( point );
            }
        }
        cloudPtr->width = (int)cloudPtr->points.size ();
        cloudPtr->height = 1;
    }

} // end ns am

#endif // DEPTHVIEWER3D_H
