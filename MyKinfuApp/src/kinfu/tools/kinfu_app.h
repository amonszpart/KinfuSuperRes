#ifndef KINFU_APP_H
#define KINFU_APP_H

#include "evaluation.h"

#include "tsdf_volume.h"
#include "tsdf_volume.hpp"

// My reverseengineered includes
#include "scene_cloud_view.h"
#include "image_view.h"
#include "point_cloud_color_handler_rgb_cloud.hpp"
#include "my_screenshot_manager.h"
#include "kinfu_util.h"

#include "kinfu_pcl_headers.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace am
{

    struct CurrentFrameCloudView
    {
            CurrentFrameCloudView() : cloud_device_ (480, 640), cloud_viewer_ ("Frame Cloud Viewer")
            {
                cloud_ptr_ = PointCloud<PointXYZ>::Ptr (new PointCloud<PointXYZ>);

                cloud_viewer_.setBackgroundColor (0, 0, 0.15);
                cloud_viewer_.setPointCloudRenderingProperties (visualization::PCL_VISUALIZER_POINT_SIZE, 1);
                cloud_viewer_.addCoordinateSystem (1.0);
                cloud_viewer_.initCameraParameters ();
                cloud_viewer_.setPosition (0, 500);
                cloud_viewer_.setSize (640, 480);
                cloud_viewer_.setCameraClipDistances (0.01, 10.01);
            }

            void
            show (const KinfuTracker& kinfu)
            {
                kinfu.getLastFrameCloud (cloud_device_);

                int c;
                cloud_device_.download (cloud_ptr_->points, c);
                cloud_ptr_->width = cloud_device_.cols ();
                cloud_ptr_->height = cloud_device_.rows ();
                cloud_ptr_->is_dense = false;

                cloud_viewer_.removeAllPointClouds ();
                cloud_viewer_.addPointCloud<PointXYZ>(cloud_ptr_);
                cloud_viewer_.spinOnce ();
            }

            void
            setViewerPose (const Eigen::Affine3f& viewer_pose) {
                ::setViewerPose (cloud_viewer_, viewer_pose);
            }

            PointCloud<PointXYZ>::Ptr cloud_ptr_;
            DeviceArray2D<PointXYZ> cloud_device_;
            visualization::PCLVisualizer cloud_viewer_;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct KinFuApp
    {
            KinFuApp(pcl::Grabber& source, float vsz, int icp, int viz);

            ~KinFuApp();

            void
            initCurrentFrameView ();

            void
            initRegistration ();

            void
            toggleColorIntegration();

            void
            enableTruncationScaling();

            void
            toggleIndependentCamera();

            void
            toggleEvaluationMode(const string& eval_folder, const string& match_file = string());

            void execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data);

            void source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper);

            void source_cb2_device(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float);


            void source_cb1_oni(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper);

            void source_cb2_oni(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float);

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            void
            startMainLoop (bool triggered_capture, int limit_frames = -1);

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            void
            writeCloud (int format, std::string fileName = "cloud" ) const;

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            void
            writeMesh( int format, std::string fileName = "mesh" );

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            void
            extractMeshFromVolume();

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            void
            saveTSDFVolume( std::string fileName = "cloud" );

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            void
            printHelp ();

            bool exit_;
            bool scan_;
            bool scan_mesh_;
            bool scan_volume_;

            bool independent_camera_;

            bool registration_;
            bool integrate_colors_;
            float focal_length_;
            bool dump_poses_;

            pcl::Grabber& capture_;
            KinfuTracker kinfu_;
            MyScreenshotManager screenshot_manager_;

            SceneCloudView scene_cloud_view_;
            ImageView image_view_;
            //ImageView rgb_view_;// aron
            boost::shared_ptr<CurrentFrameCloudView> current_frame_cloud_view_;

            KinfuTracker::DepthMap depth_device_;

            pcl::TSDFVolume<float, short> tsdf_volume_;
            pcl::PointCloud<pcl::PointXYZI>::Ptr tsdf_cloud_ptr_;

            Evaluation::Ptr evaluation_ptr_;

            boost::mutex              data_ready_mutex_;
            boost::condition_variable data_ready_cond_;

            std::vector<KinfuTracker::PixelRGB> source_image_data_;
            std::vector<unsigned short> source_depth_data_;
            PtrStepSz<const unsigned short> depth_;
            PtrStepSz<const KinfuTracker::PixelRGB> rgb24_;

            int time_ms_;
            int icp_, viz_;

            /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
            static void
            keyboard_callback (const visualization::KeyboardEvent &e, void *cookie);
    };

} // ns am

#endif // KINFU_APP_H
