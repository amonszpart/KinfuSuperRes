#include "DepthViewer3D.h"

#include "tsdf_viewer.h"

#include <pcl/common/centroid.h>

DepthViewer3D::DepthViewer3D()
    : viewer_ptr_(new pcl::visualization::PCLVisualizer("Color Cloud Viewer"))
{
    viewer_ptr_->setBackgroundColor( 0, 0, 0 );
    viewer_ptr_->addCoordinateSystem( 1.0 );
    viewer_ptr_->initCameraParameters();
    viewer_ptr_->setPosition( 0, 500 );
    viewer_ptr_->setSize( 1280, 960 );
    viewer_ptr_->setCameraClipDistances( 0.01, 10.01 );
    viewer_ptr_->setShowFPS( false );
}

void
DepthViewer3D::showMats( cv::Mat const& large_dep16, cv::Mat const& rgb8_960, int img_id, std::map<int,Eigen::Affine3f> const& poses,  Eigen::Matrix3f const& intrinsics )
{
    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloudPtr;
    DepthViewer3D::matsTo3D<ushort>( large_dep16, rgb8_960, colorCloudPtr, intrinsics );

    Eigen::Vector4f centroid;
    pcl::compute3DCentroid( *colorCloudPtr, centroid );
    std::cout << "centroid: " << centroid << " "
              << centroid.cols() << " " << centroid.rows()
              << centroid.block<3,1>(0,0).transpose() << std::endl;

    Eigen::Vector3f up = am::TSDFViewer::getViewerCameraUp( *viewer_ptr_ );
    //Eigen::Matrix3f r = Eigen::AngleAxisf( 0.25 * M_PI, up).toRotationMatrix();
    Eigen::Affine3f pose = poses.at( img_id );
    pose *= Eigen::AngleAxisf( 0.5 * M_PI, (centroid.block<3,1>(0,0) + up).normalized() );
    //pose.linear() *= r;
    am::TSDFViewer::setViewerPose( *viewer_ptr_, pose );
    //am::TSDFViewer::setViewerFovy( *pColorCloudViewer, intr_m3f );
    pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb( colorCloudPtr );
    viewer_ptr_->addPointCloud<pcl::PointXYZRGB>( colorCloudPtr, rgb, "colorCloud" );

    viewer_ptr_->spin();
}

