#include "DepthViewer3D.h"

#include "tsdf_viewer.h"

#include <vtkCamera.h>
#include <pcl/common/centroid.h>
#include <pcl/common/common.h>

#include "AmPclUtil.h"

namespace am
{

    void depth_viewer3d_keyboard_callback( const pcl::visualization::KeyboardEvent &e, void *cookie )
    {
        pcl::visualization::PCLVisualizer::Ptr* pViewerPtr = reinterpret_cast<pcl::visualization::PCLVisualizer::Ptr*>( cookie );

        int key = e.getKeyCode ();
        Eigen::Affine3f tmp_pose = (*pViewerPtr)->getViewerPose();
        if ( e.keyUp () )
        {
            switch ( key )
            {
                case 27:
                    //(*pViewerPtr)->
                    break;

                case 82:
                case 'a':
                    tmp_pose.linear() *= Eigen::AngleAxisf( .25f * M_PI, Eigen::Vector3f::UnitY() ).matrix();
                    break;

                case 'd':
                case 83:
                    tmp_pose.linear() *= Eigen::AngleAxisf( -.25f * M_PI, Eigen::Vector3f::UnitY() ).matrix();
                    break;

                case 'w':
                case 81:
                    tmp_pose.linear() *= Eigen::AngleAxisf( .25f * M_PI, Eigen::Vector3f::UnitX() ).matrix();
                    break;

                case 's':
                case 84:
                    tmp_pose.linear() *= Eigen::AngleAxisf( -.25f * M_PI, Eigen::Vector3f::UnitX() ).matrix();
                    break;

                case 't':
                    tmp_pose.linear() *= Eigen::AngleAxisf( .25f * M_PI, Eigen::Vector3f::UnitZ() ).matrix();
                    break;

                case 'g':
                    tmp_pose.linear() *= Eigen::AngleAxisf( -.25f * M_PI, Eigen::Vector3f::UnitZ() ).matrix();

                    break;

                default:
                    break;
            }
            am::util::pcl::setViewerPose( **pViewerPtr, tmp_pose );
            am::util::pcl::printPose( tmp_pose );
        }
        std::cout << "keypress: " << (int)key << std::endl;
    }

    void depth_viewer3d_mouse_callback (const pcl::visualization::MouseEvent& mouse_event, void* cookie)
    {
        // player pointer
        DepthViewer3D *p_depthViewer = reinterpret_cast<DepthViewer3D*>( cookie );

        // left button release
        if ( mouse_event.getType()   == pcl::visualization::MouseEvent::MouseButtonRelease &&
             mouse_event.getButton() == pcl::visualization::MouseEvent::LeftButton            )
        {
            // read
            Eigen::Affine3f tmp_pose = p_depthViewer->ViewerPtr()->getViewerPose();
            am::util::pcl::printPose( tmp_pose );

            p_depthViewer->broadcastPoseOf( p_depthViewer->ViewerPtr() );
        }
    }

    DepthViewer3D::DepthViewer3D()
        : viewer_ptr_(new pcl::visualization::PCLVisualizer("Color Cloud Viewer"))
    {
        viewer_ptr_->setBackgroundColor( 0, 0, 0 );
        viewer_ptr_->setPosition( 0, 500 );
        viewer_ptr_->initCameraParameters();
        viewer_ptr_->setSize( 1280, 960 );
        viewer_ptr_->setCameraClipDistances( 0.01, 10.01 );
        viewer_ptr_->setShowFPS( true );
        viewer_ptr_->addCoordinateSystem(1.f, 0, 0, 0 );
//        viewer_ptr_->addCoordinateSystem(5.f, 0, 1, 0 );
//        viewer_ptr_->addCoordinateSystem(3.f, 0, 0, 1 );
//        viewer_ptr_->addCoordinateSystem(1.f, 0, 0, 0 );
        viewer_ptr_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3 );

        //boost::function<void (const pcl::visualization::KeyboardEvent&, void*)> func_keyboard = boost::bind (&DepthViewer3D::keyboard_callback, this, _1, _2 );
        viewer_ptr_->registerKeyboardCallback( depth_viewer3d_keyboard_callback, (void*)&viewer_ptr_ );
        viewer_ptr_->registerMouseCallback   ( depth_viewer3d_mouse_callback   , (void*)this );
    }

    /*
     * \brief Depth image (and registered colour) to point cloud
     */
    void
    DepthViewer3D::showMats( cv::Mat const& large_dep16, cv::Mat const& rgb8_960, int img_id, std::map<int,Eigen::Affine3f> const& poses, Eigen::Matrix3f const& intrinsics )
    {
        // create cloud
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloudPtr;
        {
            DepthViewer3D::matsTo3D<ushort>( large_dep16, rgb8_960, cloudPtr, intrinsics, 1.f / 1000.f, false, &(poses.at(img_id)) );
        }

        // calculate centroid
        Eigen::Vector4f centroid;
        {
            pcl::compute3DCentroid( *cloudPtr, centroid );
            std::cout << "centroid: " << centroid.transpose() << " "
                      << centroid.cols() << " " << centroid.rows()
                      << centroid.block<3,1>(0,0).transpose() << std::endl;
        }

        // bounding box
        Eigen::Vector4f min_pt, max_pt;
        {
            pcl::getMinMax3D( *cloudPtr, min_pt, max_pt );
            std::cout << "min_pt: " << min_pt.transpose() << std::endl
                      << "max_pt: " << max_pt.transpose() << std::endl;
        }
#if 0
        // set camera
        {
            Eigen::Vector3f up = am::TSDFViewer::getViewerCameraUp( *viewer_ptr_ );
            Eigen::Affine3f pose = poses.at( img_id );
            pose *= Eigen::AngleAxisf( 0.5 * M_PI, (centroid.block<3,1>(0,0) + up).normalized() );
            am::TSDFViewer::setViewerPose( *viewer_ptr_, pose );
            //am::TSDFViewer::setViewerFovy( *pColorCloudViewer, intr_m3f );
        }
#endif

        // show cloud
        {
            pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb( cloudPtr );
            viewer_ptr_->addPointCloud<pcl::PointXYZRGB>( cloudPtr, rgb, "colorCloud" );

            //viewer_ptr_->spinOnce();
            //viewer_ptr_->showCl
            //viewer_ptr_->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "colorCloud" );
        }

        // set camera
        {
            Eigen::Vector3f camPos;
            camPos[0] = min_pt[0];// + (max_pt[0] - min_pt[0]) / 4.f; // max at bin
            camPos[1] = min_pt[1];// + (max_pt[1] - min_pt[1]) / 6.f; // max at left corner
            camPos[2] = min_pt[2] + (max_pt[2] - min_pt[2]) / 4.f; // max bottom of floor
            Eigen::Affine3f pose;
            //pose.linear() = (//Eigen::AngleAxisf(0.f * M_PI,Eigen::Vector3f::UnitX()) *
            //                 Eigen::AngleAxisf(.5f * M_PI,Eigen::Vector3f::UnitY()) *
            //                 Eigen::AngleAxisf(.5f * M_PI,Eigen::Vector3f::UnitZ())  ).matrix();
            pose.translation() = camPos;
            //am::TSDFViewer::setViewerPose( *viewer_ptr_, pose );
            /*viewer_ptr_->setCameraPosition( .4,0.6, -1.,//camPos[0], camPos[1], camPos[2],
                                            //centroid[0], centroid[1], centroid[2],
                                            2,2,2,
                                            .0, .0, -1. );*/
            //viewer_ptr_->addCoordinateSystem(3.f, camPos[0], camPos[1], camPos[2] );
            //viewer_ptr_->addCoordinateSystem(2.f, -1.f, 1.f, 1.f );

            /*Eigen::Matrix4f pose2;
            pose2 << 0.369991, 0.318449, -0.872752, 1.25756,
                     -0.928471,0.15949, -0.335418, .572008,
                     0.032382, 0.934426,  0.354681, .0893645,
                     .0       ,        .0,        .0,      1.;*/

            //am::util::pcl::printPose( viewer_ptr_->getViewerPose() );
            //am::util::pcl::setViewerPose( *viewer_ptr_, pose2 );
            am::util::pcl::printPose( viewer_ptr_->getViewerPose() );

        }

        viewer_ptr_->spinOnce();
    }

    void
    DepthViewer3D::showAllPoses()
    {
        am::DepthViewer3D depthViewer;
        //depthViewer.showMats( large_dep16, rgb8_960, img_id, poses, intrinsics );
    }
} // end ns am

