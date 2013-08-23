#include "AmPclUtil.h"

#include <vtkRenderWindow.h>
#include <vtkWindowToImageFilter.h>
#include <vtkImageShiftScale.h>
#include <vtkCamera.h>
#include <vtkImageData.h>
#include <vtkSmartPointer.h>

#include <iostream>

namespace am
{
    namespace util
    {
        namespace pcl
        {
            void fetchViewerZBuffer( /* out: */ cv::Mat & zBufMat,
                                     /*  in: */ ::pcl::visualization::PCLVisualizer::Ptr const& viewer, double zNear, double zFar )
            {
                std::cout << "saving vtkZBuffer...";

                vtkSmartPointer<vtkWindowToImageFilter> filter  = vtkSmartPointer<vtkWindowToImageFilter>::New();
                vtkSmartPointer<vtkImageShiftScale>     scale   = vtkSmartPointer<vtkImageShiftScale>::New();
                vtkSmartPointer<vtkWindow>              renWin  = viewer->getRenderWindow();

                // Create Depth Map
                filter->SetInput( renWin.GetPointer() );
                filter->SetMagnification(1);
                filter->SetInputBufferTypeToZBuffer();

                // scale
                scale->SetOutputScalarTypeToFloat();
                scale->SetInputConnection(filter->GetOutputPort());
                scale->SetShift(0.f);
                scale->SetScale(1.f);
                scale->Update();

                // fetch data
                vtkSmartPointer<vtkImageData> imageData = scale->GetOutput();
                int* dims = imageData->GetDimensions();
                if ( dims[2] > 1 )
                {
                    std::cerr << "am::util::pcl::fetchZBuffer(): ZDim != 1 !!!!" << std::endl;
                }

                // output
                zBufMat.create( dims[1], dims[0], CV_16UC1 );
                for ( int y = 0; y < dims[1]; ++y )
                {
                    for ( int x = 0; x < dims[0]; ++x )
                    {
                        float* pixel = static_cast<float*>( imageData->GetScalarPointer(x,y,0) );
                        ushort d = round(2.0 * zNear * zFar / (zFar + zNear - pixel[0] * (zFar - zNear)) * 1000.f);
                        zBufMat.at<ushort>( dims[1] - y - 1, x ) = (d > 10001) ? 0 : d;

                        //data[ z * dims[1] * dims[0] + (dims[1] - y - 1) * dims[0] + x ] = pixel[0];//(pixel[0] == 10001) ? 0 : pixel[0];
                    }
                    //std::cout << std::endl;
                }
                //std::cout << std::endl;
            }

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

            void
            printPose( Eigen::Affine3f const& pose )
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
