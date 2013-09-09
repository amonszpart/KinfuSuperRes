#ifndef DEPTHVIEWER3D_H
#define DEPTHVIEWER3D_H

#include "ViewPointMapperCuda.h"

#include <pcl/visualization/pcl_visualizer.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "AmPclUtil.h"

#include <map>

namespace am
{

    class CameraPoseBroadcaster
    {
        public:
            void addListener( pcl::visualization::PCLVisualizer::Ptr const& vis )
            {
                visualizers_.push_back( vis );
            }

            void broadcastPoseOf( pcl::visualization::PCLVisualizer::Ptr const& vis_arg )
            {
                Eigen::Vector3d pos, up, dir;
                am::util::pcl::getCam( pos, up, dir, vis_arg );
                for ( auto &visualizer : visualizers_ )
                {
                    am::util::pcl::setCam( pos, up, dir, visualizer );
                }
            }

        protected:
            std::vector<pcl::visualization::PCLVisualizer::Ptr> visualizers_;

    };

    class DepthViewer3D : public CameraPoseBroadcaster
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
                                  Eigen::Affine3f const* const pose,
                                  std::vector< ::pcl::Vertices> *pFaces = NULL ); /* display scaling of pointcloud */
            static void
            showAllPoses();

            pcl::visualization::PCLVisualizer::Ptr ViewerPtr() { return viewer_ptr_; }
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr CloudPtr()  { return cloud_ptr_;  }

        protected:
            // FIELDS
            pcl::visualization::PCLVisualizer::Ptr viewer_ptr_;
            pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_ptr_;

    };

    template <typename depT>
    void
    DepthViewer3D::matsTo3D( cv::Mat const& dep, cv::Mat const& img,
                             pcl::PointCloud<pcl::PointXYZRGB>::Ptr & cloudPtr,
                             Eigen::Matrix3f const& intrinsics,
                             /* depth scale: */ float alpha, bool useColour,
                             Eigen::Affine3f const* const pose,
                             std::vector< ::pcl::Vertices> *pFaces )
    {
        cv::Mat imgResized;
        // check input
        if ( !img.empty() )
        {
            if ( dep.size() != img.size() )
            {
                std::cerr << "matsTo3D(): dep and rgb need the same size! "
                          << dep.rows << "x" << dep.cols << ", "
                          << img.rows << "x" << img.cols << std::endl;
                std::cerr << "resizing color..." << std::endl;
                cv::resize( img, imgResized, dep.size(), 0, 0, cv::INTER_LANCZOS4 );
            }
            else
            {
                img.copyTo( imgResized );
            }
        }

        if ( (!img.empty()) && (img.channels() != 3) )
        {
            std::cerr << "matsTo3D(): rgb should be 3 channels! " << img.channels() << std::endl;
            return;
        }

#if 0
        {
            double minVal, maxVal;
            cv::minMaxIdx( dep, &minVal, &maxVal );
            if      ( maxVal > 1000.f ) { std::cout << "matsTo3D(): dividing by 1000.f" << std::endl; dep /= 1000.f; }
            else if ( maxVal > 100.f  ) { std::cout << "matsTo3D(): dividing by 100.f"  << std::endl; dep /= 100.f;  }
            else if ( maxVal > 10.f   ) { std::cout << "matsTo3D(): dividing by 10.f"   << std::endl; dep /= 10.f;   }
        }
#endif


        // allocate ouptut
        cloudPtr = pcl::PointCloud<pcl::PointXYZRGB>::Ptr( new pcl::PointCloud<pcl::PointXYZRGB> );

        Eigen::Vector3f pnt3D;
        uint32_t rgb = 128 << 16 | 128 << 8 | 128;

        // test
        float *pixMap = new float[ dep.cols * dep.rows * 2 ];
        ViewPointMapperCuda::runCam2World( pixMap, dep.cols, dep.rows,
                                           /* fx: */ intrinsics(0,0), /* fy: */ intrinsics(1,1),
                                           /* cx: */ intrinsics(0,2), /* cy: */ intrinsics(1,2),
                                           0.f, 0.f, 0.f, 0.f, 0.f, 0.f );
        Eigen::Matrix3f rotation;
        Eigen::Vector3f translation;
        if ( pose )
        {
            rotation = pose->rotation();
            translation = pose->translation();
        }

        if ( pFaces )
        {
            pFaces->clear();
            pFaces->reserve( dep.rows * dep.cols * 2 );
        }

        int fid = 0;
        // copy inputs
        for ( int y = 0; y < dep.rows; ++y )
        {
            for ( int x = 0; x < dep.cols; ++x )
            {
#if 0
                pnt3D = am::util::pcl::point2To3D( (Eigen::Vector2f){x,y},
                                                   intrinsics            )
                        * (float)dep.at<depT>( y,x ) * alpha;
                /*if (     ( pnt3D(0) != pixMap[ (y * dep.cols + x) * 2    ] )
                      || ( pnt3D(1) != pixMap[ (y * dep.cols + x) * 2 + 1] )  )
                {
                    std::cout << "pnt3D: "
                              << pnt3D(0) << "," << pnt3D(1)
                              << " != "
                              << pixMap[ (y * dep.cols + x) * 2     ] << ","
                              << pixMap[ (y * dep.cols + x) * 2 + 1 ]
                              << std::endl;
                }*/
#else
                pnt3D = (Eigen::Vector3f)
                { pixMap[ (y * dep.cols + x) * 2     ],
                  pixMap[ (y * dep.cols + x) * 2 + 1 ],
                  1.f } * ( (float)dep.at<depT>( y,x ) * alpha );
#endif
                if ( pose )
                {
                    pnt3D = rotation * pnt3D + translation;
                }

                pcl::PointXYZRGB point;
                point.x = pnt3D(0); //point.x = (x - intrinsics(0,2)) / intrinsics(0,0) * dep.at<depT>( y,x ) * alpha;
                point.y = pnt3D(1); //point.y = (y - intrinsics(1,2)) / intrinsics(1,1) * dep.at<depT>( y,x ) * alpha;
                point.z = pnt3D(2); //point.z = dep.at<depT>( y,x ) * alpha;

                if ( !imgResized.empty() )
                {
                    rgb = (static_cast<uint32_t>(imgResized.at<uchar>(y, x * imgResized.channels() + 2)) << 16 |
                           static_cast<uint32_t>(imgResized.at<uchar>(y, x * imgResized.channels() + 1)) << 8  |
                           static_cast<uint32_t>(imgResized.at<uchar>(y, x * imgResized.channels()    ))        );
                }

                point.rgb = *reinterpret_cast<float*>( &rgb );
                cloudPtr->points.push_back( point );

                // faces
                if ( (pFaces) && ((float)dep.at<depT>( y,x ) > 0.f) && (y+1 < dep.rows) )
                {
                    if ( x > 0 ) // left face
                    {
                        pFaces->push_back( ::pcl::Vertices() );
                        pFaces->back().vertices.resize(3);
                        pFaces->back().vertices[0] =  y    * dep.cols + x    ; // y,x
                        pFaces->back().vertices[1] = (y+1) * dep.cols + x - 1; // y+1,x-1
                        pFaces->back().vertices[2] = (y+1) * dep.cols + x    ; // y+1,x

                        //++fid;
                    }
                    if ( x+1 < dep.cols ) // right face
                    {
                        pFaces->push_back( ::pcl::Vertices() );
                        pFaces->back().vertices.resize(3);
                        pFaces->back().vertices[0] =  y    * dep.cols + x    ; // y,x
                        pFaces->back().vertices[1] = (y+1) * dep.cols + x    ; // y+1,x
                        pFaces->back().vertices[2] =  y    * dep.cols + x + 1; // y+1,x+1

                        //++fid;
                    }
                }
            }
        }
        cloudPtr->width = (int)cloudPtr->points.size ();
        cloudPtr->height = 1;

        delete [] pixMap;
    }

} // end ns am

#endif // DEPTHVIEWER3D_H
