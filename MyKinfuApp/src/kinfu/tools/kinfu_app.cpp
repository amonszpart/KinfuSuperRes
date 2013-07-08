/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Anatoly Baskeheev, Itseez Ltd, (myname.mysurname@mycompany.com)
 */
#define _CRT_SECURE_NO_DEPRECATE

#include "kinfu_app.h"

#include <pcl/common/angles.h>
//#include "../src/internal.h"



#include <iostream>

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace am
{
    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    struct SampledScopeTime : public StopWatch
    {
            enum { EACH = 33 };
            SampledScopeTime(int& time_ms) : time_ms_(time_ms) {}
            ~SampledScopeTime()
            {
                static int i_ = 0;
                time_ms_ += getTime ();
                if (i_ % EACH == 0 && i_)
                {
                    cout << "Average frame time = " << time_ms_ / EACH << "ms ( " << 1000.f * EACH / time_ms_ << "fps )" << endl;
                    time_ms_ = 0;
                }
                ++i_;
            }
        private:
            int& time_ms_;
    };

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    KinFuApp::KinFuApp(pcl::Grabber& source, float vsz, int icp, int viz)
        : exit_ (false), scan_ (false), scan_mesh_(false), scan_volume_ (false), independent_camera_ (false),
          registration_ (false), integrate_colors_ (false), dump_poses_ (false), focal_length_(-1.f), capture_ (source), scene_cloud_view_(viz), image_view_(viz), time_ms_(0), icp_(icp), viz_(viz)
    {
        //Init Kinfu Tracker
        Eigen::Vector3f volume_size = Vector3f::Constant (vsz/*meters*/);
        kinfu_.volume().setSize (volume_size);

        Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
        Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

        Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

        kinfu_.setInitalCameraPose (pose);
        kinfu_.volume().setTsdfTruncDist (0.030f/*meters*/);
        kinfu_.setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
        //kinfu_.setDepthTruncationForICP(5.f/*meters*/);
        kinfu_.setCameraMovementThreshold(0.001f);

        if (!icp)
            kinfu_.disableIcp();


        //Init KinfuApp
        tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZI>::Ptr (new pcl::PointCloud<pcl::PointXYZI>);
        image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols ()) );

        if (viz_)
        {
            scene_cloud_view_.cloud_viewer_->registerKeyboardCallback (keyboard_callback, (void*)this);
            image_view_.viewerScene_->registerKeyboardCallback (keyboard_callback, (void*)this);
            image_view_.viewerDepth_->registerKeyboardCallback (keyboard_callback, (void*)this);

            scene_cloud_view_.toggleCube(volume_size);
        }

        // init dumping
        if ( dump_poses_ )
        {
            screenshot_manager_.setCameraIntrinsics( focal_length_, 640, 480 );
        }
    }

    KinFuApp::~KinFuApp()
    {
        if (evaluation_ptr_)
            evaluation_ptr_->saveAllPoses(kinfu_);
    }

    void
    KinFuApp::initCurrentFrameView ()
    {
        current_frame_cloud_view_ = boost::shared_ptr<CurrentFrameCloudView>(new CurrentFrameCloudView ());
        current_frame_cloud_view_->cloud_viewer_.registerKeyboardCallback (keyboard_callback, (void*)this);
        current_frame_cloud_view_->setViewerPose (kinfu_.getCameraPose ());
    }

    void
    KinFuApp::initRegistration ()
    {
        registration_ = capture_.providesCallback<pcl::ONIGrabber::sig_cb_openni_image_depth_image> ();
        cout << "Registration mode: " << (registration_ ? "On" : "Off (not supported by source)") << endl;
    }

    void
    KinFuApp::toggleColorIntegration()
    {
        if(registration_)
        {
            const int max_color_integration_weight = 2;
            kinfu_.initColorIntegration(max_color_integration_weight);
            integrate_colors_ = true;
        }
        cout << "Color integration: " << (integrate_colors_ ? "On" : "Off ( requires registration mode )") << endl;
    }

    void
    KinFuApp::enableTruncationScaling()
    {
        kinfu_.volume().setTsdfTruncDist (kinfu_.volume().getSize()(0) / 100.0f);
    }

    void
    KinFuApp::toggleIndependentCamera()
    {
        independent_camera_ = !independent_camera_;
        cout << "Camera mode: " << (independent_camera_ ?  "Independent" : "Bound to Kinect pose") << endl;
    }

    void
    KinFuApp::toggleEvaluationMode(const string& eval_folder, const string& match_file )
    {
        evaluation_ptr_ = Evaluation::Ptr( new Evaluation(eval_folder) );
        if (!match_file.empty())
            evaluation_ptr_->setMatchFile(match_file);

        kinfu_.setDepthIntrinsics (evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy);
        image_view_.raycaster_ptr_ = RayCaster::Ptr( new RayCaster(kinfu_.rows (), kinfu_.cols (),
                                                                   evaluation_ptr_->fx, evaluation_ptr_->fy, evaluation_ptr_->cx, evaluation_ptr_->cy) );
    }

    void
    KinFuApp::source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)
    {
        {
            boost::mutex::scoped_try_lock lock(data_ready_mutex_);
            if (exit_ || !lock)
                return;

            depth_.cols = depth_wrapper->getWidth();
            depth_.rows = depth_wrapper->getHeight();
            depth_.step = depth_.cols * depth_.elemSize();

            source_depth_data_.resize(depth_.cols * depth_.rows);
            depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
            depth_.data = &source_depth_data_[0];
        }
        data_ready_cond_.notify_one();
    }

    void
    KinFuApp::source_cb2_device(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
    {
        {
            boost::mutex::scoped_try_lock lock(data_ready_mutex_);
            if (exit_ || !lock)
                return;

            depth_.cols = depth_wrapper->getWidth();
            depth_.rows = depth_wrapper->getHeight();
            depth_.step = depth_.cols * depth_.elemSize();

            source_depth_data_.resize(depth_.cols * depth_.rows);
            depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
            depth_.data = &source_depth_data_[0];

            rgb24_.cols = image_wrapper->getWidth();
            rgb24_.rows = image_wrapper->getHeight();
            rgb24_.step = rgb24_.cols * rgb24_.elemSize();

            source_image_data_.resize(rgb24_.cols * rgb24_.rows);
            image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
            rgb24_.data = &source_image_data_[0];
        }
        data_ready_cond_.notify_one();
    }

    void
    KinFuApp::source_cb1_oni(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)
    {
        {
            boost::mutex::scoped_lock lock(data_ready_mutex_);
            if (exit_)
                return;

            depth_.cols = depth_wrapper->getWidth();
            depth_.rows = depth_wrapper->getHeight();
            depth_.step = depth_.cols * depth_.elemSize();

            source_depth_data_.resize(depth_.cols * depth_.rows);
            depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
            depth_.data = &source_depth_data_[0];
        }
        data_ready_cond_.notify_one();
    }

    void
    KinFuApp::source_cb2_oni(const boost::shared_ptr<openni_wrapper::Image>& image_wrapper, const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper, float)
    {
        {
            boost::mutex::scoped_lock lock(data_ready_mutex_);
            if (exit_)
                return;

            depth_.cols = depth_wrapper->getWidth();
            depth_.rows = depth_wrapper->getHeight();
            depth_.step = depth_.cols * depth_.elemSize();

            source_depth_data_.resize(depth_.cols * depth_.rows);
            depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
            depth_.data = &source_depth_data_[0];

            rgb24_.cols = image_wrapper->getWidth();
            rgb24_.rows = image_wrapper->getHeight();
            rgb24_.step = rgb24_.cols * rgb24_.elemSize();

            source_image_data_.resize(rgb24_.cols * rgb24_.rows);
            image_wrapper->fillRGB(rgb24_.cols, rgb24_.rows, (unsigned char*)&source_image_data_[0]);
            rgb24_.data = &source_image_data_[0];
        }
        data_ready_cond_.notify_one();
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::writeCloud (int format, std::string fileName ) const
    {
        const SceneCloudView& view = scene_cloud_view_;

        if(view.point_colors_ptr_->points.empty()) // no colors
        {
            if (view.valid_combined_)
                writeCloudFile (format, view.combined_ptr_, fileName);
            else
                writeCloudFile (format, view.cloud_ptr_, fileName);
        }
        else
        {
            if (view.valid_combined_)
                writeCloudFile (format, merge<PointXYZRGBNormal>(*view.combined_ptr_, *view.point_colors_ptr_), fileName);
            else
                writeCloudFile (format, merge<PointXYZRGB>(*view.cloud_ptr_, *view.point_colors_ptr_), fileName);
        }
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::writeMesh( int format, std::string fileName )
    {
        if ( !scene_cloud_view_.mesh_ptr_ )
        {
            std::cout << "scene_cloud_view_.mesh_ptr_ is empty, so extracting mesh..." << std::endl;
            extractMeshFromVolume ();
        }
        writePolygonMeshFile(format, *scene_cloud_view_.mesh_ptr_, fileName );
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::extractMeshFromVolume()
    {
        ScopeTimeT time ( "Mesh Extraction" );
        cout << "\nGetting mesh... " << flush;

        if ( !scene_cloud_view_.marching_cubes_ )
            scene_cloud_view_.marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

        DeviceArray<PointXYZ> triangles_device = scene_cloud_view_.marching_cubes_->run(kinfu_.volume(), scene_cloud_view_.triangles_buffer_device_);
        scene_cloud_view_.mesh_ptr_ = convertToMesh( triangles_device );
    }

    void
    KinFuApp::saveTSDFVolume( std::string fileName )
    {
        cout << "Saving TSDF volume to " + fileName + "_tsdf_volume.dat ... " << flush;
        this->tsdf_volume_.save ( fileName + "_tsdf_volume.dat", true );
        cout << "done [" << this->tsdf_volume_.size () << " voxels]" << endl;

        cout << "Saving TSDF volume cloud to " + fileName + "_tsdf_cloud.pcd ... " << flush;
        pcl::io::savePCDFile<pcl::PointXYZI> (fileName+"_tsdf_cloud.pcd", *this->tsdf_cloud_ptr_, true);
        cout << "done [" << this->tsdf_cloud_ptr_->size () << " points]" << endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::printHelp ()
    {
        cout << endl;
        cout << "KinFu app hotkeys" << endl;
        cout << "=================" << endl;
        cout << "    H    : print this help" << endl;
        cout << "   Esc   : exit" << endl;
        cout << "    T    : take cloud" << endl;
        cout << "    A    : take mesh" << endl;
        cout << "    M    : toggle cloud exctraction mode" << endl;
        cout << "    N    : toggle normals exctraction" << endl;
        cout << "    I    : toggle independent camera mode" << endl;
        cout << "    B    : toggle volume bounds" << endl;
        cout << "    *    : toggle scene view painting ( requires registration mode )" << endl;
        cout << "    C    : clear clouds" << endl;
        cout << "   1,2,3 : save cloud to PCD(binary), PCD(ASCII), PLY(ASCII)" << endl;
        cout << "    7,8  : save mesh to PLY, VTK" << endl;
        cout << "   X, V  : TSDF volume utility" << endl;
        cout << endl;
    }

    /////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::keyboard_callback (const visualization::KeyboardEvent &e, void *cookie)
    {
        KinFuApp* app = reinterpret_cast<KinFuApp*> (cookie);

        int key = e.getKeyCode ();

        if (e.keyUp ())
            switch (key)
            {
                case 27: app->exit_ = true; break;
                case (int)'t': case (int)'T': app->scan_ = true; break;
                case (int)'a': case (int)'A': app->scan_mesh_ = true; break;
                case (int)'h': case (int)'H': app->printHelp (); break;
                case (int)'m': case (int)'M': app->scene_cloud_view_.toggleExtractionMode (); break;
                case (int)'n': case (int)'N': app->scene_cloud_view_.toggleNormals (); break;
                case (int)'c': case (int)'C': app->scene_cloud_view_.clearClouds (true); break;
                case (int)'i': case (int)'I': app->toggleIndependentCamera (); break;
                case (int)'b': case (int)'B': app->scene_cloud_view_.toggleCube(app->kinfu_.volume().getSize()); break;
                case (int)'7': case (int)'8': app->writeMesh (key - (int)'0'); break;
                case (int)'1': case (int)'2': case (int)'3': app->writeCloud (key - (int)'0'); break;
                case '*': app->image_view_.toggleImagePaint (); break;

                case (int)'x': case (int)'X':
                    app->scan_volume_ = !app->scan_volume_;
                    cout << endl << "Volume scan: " << (app->scan_volume_ ? "enabled" : "disabled") << endl << endl;
                    break;
                case (int)'v': case (int)'V':
                    cout << "Saving TSDF volume to tsdf_volume.dat ... " << flush;
                    app->tsdf_volume_.save ("tsdf_volume.dat", true);
                    cout << "done [" << app->tsdf_volume_.size () << " voxels]" << endl;
                    cout << "Saving TSDF volume cloud to tsdf_cloud.pcd ... " << flush;
                    pcl::io::savePCDFile<pcl::PointXYZI> ("tsdf_cloud.pcd", *app->tsdf_cloud_ptr_, true);
                    cout << "done [" << app->tsdf_cloud_ptr_->size () << " points]" << endl;
                    break;

                default:
                    break;
            }
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    int
    print_cli_help ()
    {
        cout << "\nKinFu parameters:" << endl;
        cout << "    --help, -h                      : print this message" << endl;
        cout << "    --registration, -r              : try to enable registration (source needs to support this)" << endl;
        cout << "    --current-cloud, -cc            : show current frame cloud" << endl;
        cout << "    --save-views, -sv               : accumulate scene view and save in the end ( Requires OpenCV. Will cause 'bad_alloc' after some time )" << endl;
        cout << "    --integrate-colors, -ic         : enable color integration mode (allows to get cloud with colors)" << endl;
        cout << "    --scale-truncation, -st         : scale the truncation distance and raycaster based on the volume size" << endl;
        cout << "    -volume_size <size_in_meters>   : define integration volume size" << endl;
        cout << "Valid depth data sources:" << endl;
        cout << "    -dev <device> (default), -oni <oni_file>, -pcd <pcd_file or directory>" << endl;
        cout << "";
        cout << " For RGBD benchmark (Requires OpenCV):" << endl;
        cout << "    -eval <eval_folder> [-match_file <associations_file_in_the_folder>]" << endl;

        return 0;
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::execute(const PtrStepSz<const unsigned short>& depth, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool has_data)
    {
        bool has_image = false;

        if (has_data)
        {
            // upload depth
            depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
            // upload rgb
            if (integrate_colors_)
                image_view_.colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);

            // run Kinfu
            {
                SampledScopeTime fps(time_ms_);

                //run kinfu algorithm
                if (integrate_colors_)
                    has_image = kinfu_ (depth_device_, image_view_.colors_device_);
                else
                    has_image = kinfu_ (depth_device_);
            }

            image_view_.showDepth (depth);
            //image_view_.showGeneratedDepth(kinfu_, kinfu_.getCameraPose());
        }

        if (scan_)
        {
            scan_ = false;
            scene_cloud_view_.show (kinfu_, integrate_colors_);

            if (scan_volume_)
            {
                cout << "Downloading TSDF volume from device ... " << flush;
                kinfu_.volume().downloadTsdfAndWeighs (tsdf_volume_.volumeWriteable (), tsdf_volume_.weightsWriteable ());
                tsdf_volume_.setHeader (Eigen::Vector3i (pcl::device::VOLUME_X, pcl::device::VOLUME_Y, pcl::device::VOLUME_Z), kinfu_.volume().getSize ());
                cout << "done [" << tsdf_volume_.size () << " voxels]" << endl << endl;

                cout << "Converting volume to TSDF cloud ... " << flush;
                tsdf_volume_.convertToTsdfCloud (tsdf_cloud_ptr_);
                cout << "done [" << tsdf_cloud_ptr_->size () << " points]" << endl << endl;
            }
            else
                cout << "[!] tsdf volume download is disabled" << endl << endl;
        }

        if (scan_mesh_)
        {
            scan_mesh_ = false;
            scene_cloud_view_.showMesh(kinfu_, integrate_colors_);
        }

        if ( has_image && (scene_cloud_view_.cloud_viewer_) )
        {
            Eigen::Affine3f viewer_pose = getViewerPose(*scene_cloud_view_.cloud_viewer_);
            image_view_.showScene (kinfu_, rgb24, registration_, independent_camera_ ? &viewer_pose : 0);
        }

        if (current_frame_cloud_view_)
            current_frame_cloud_view_->show (kinfu_);

        if ( (!independent_camera_) &&
             (scene_cloud_view_.cloud_viewer_) /*aron:my addition */
             )
            setViewerPose (*scene_cloud_view_.cloud_viewer_, kinfu_.getCameraPose());

        // save screenshots and poses
        if ( dump_poses_ )
        {
            screenshot_manager_.saveImage( kinfu_.getCameraPose(), rgb24, depth );
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    void
    KinFuApp::startMainLoop (bool triggered_capture)
    {
        using namespace openni_wrapper;
        typedef boost::shared_ptr<DepthImage> DepthImagePtr;
        typedef boost::shared_ptr<Image> ImagePtr;

        boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1_dev = boost::bind (&KinFuApp::source_cb2_device, this, _1, _2, _3);
        boost::function<void (const DepthImagePtr&)> func2_dev = boost::bind (&KinFuApp::source_cb1_device, this, _1);

        boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1_oni = boost::bind (&KinFuApp::source_cb2_oni, this, _1, _2, _3);
        boost::function<void (const DepthImagePtr&)> func2_oni = boost::bind (&KinFuApp::source_cb1_oni, this, _1);

        bool is_oni = dynamic_cast<pcl::ONIGrabber*>(&capture_) != 0;
        boost::function<void (const ImagePtr&, const DepthImagePtr&, float constant)> func1 = is_oni ? func1_oni : func1_dev;
        boost::function<void (const DepthImagePtr&)> func2 = is_oni ? func2_oni : func2_dev;

        bool need_colors = integrate_colors_ || registration_ || 1;
        boost::signals2::connection c = need_colors ? capture_.registerCallback (func1) : capture_.registerCallback (func2);

        {
            boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

            if (!triggered_capture)
                capture_.start (); // Start stream

            bool scene_view_not_stopped= viz_ ? !scene_cloud_view_.cloud_viewer_->wasStopped () : true;
            bool image_view_not_stopped= viz_ ? !image_view_.viewerScene_->wasStopped () : true;

            int latest_has_data_frame = 0;
            int frame_count = 0;
            while (!exit_ && scene_view_not_stopped && image_view_not_stopped)
            {
                if (triggered_capture)
                    capture_.start(); // Triggers new frame
                bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(100));

                try { this->execute (depth_, rgb24_, has_data); }
                catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; break; }
                catch (const std::exception& /*e*/) { cout << "Exception" << endl; break; }

                if (viz_)
                    scene_cloud_view_.cloud_viewer_->spinOnce (3);

                if ( has_data )
                    latest_has_data_frame = frame_count;

                if ( frame_count - latest_has_data_frame > 9 )
                {
                    scan_ = true;
                }
                if ( frame_count - latest_has_data_frame > 10 )
                {
                    exit_ = true;
                }

                std::cout << "frame_id: " << frame_count++ << " has_data: " << has_data << std::endl;
            }

            if (!triggered_capture)
                capture_.stop (); // Stop stream
        }
        c.disconnect();
    }


    int
    mainKinfuApp (int argc, char* argv[])
    {
        if (pc::find_switch (argc, argv, "--help") || pc::find_switch (argc, argv, "-h"))
            return print_cli_help ();

        int device = 0;
        pc::parse_argument (argc, argv, "-gpu", device);
        pcl::gpu::setDevice (device);
        pcl::gpu::printShortCudaDeviceInfo (device);

        //  if (checkIfPreFermiGPU(device))
        //    return cout << endl << "Kinfu is supported only for Fermi and Kepler arhitectures. It is not even compiled for pre-Fermi by default. Exiting..." << endl, 1;

        boost::shared_ptr<pcl::Grabber> capture;

        bool triggered_capture = false;

        std::string eval_folder, match_file, openni_device, oni_file, pcd_dir;
        try
        {
            if (pc::parse_argument (argc, argv, "-dev", openni_device) > 0)
            {
                capture.reset (new pcl::OpenNIGrabber (openni_device));
            }
            else if (pc::parse_argument (argc, argv, "-oni", oni_file) > 0)
            {
                triggered_capture = true;
                bool repeat = false; // Only run ONI file once
                std::cout << "oni: " << oni_file << std::endl;
                capture.reset (new pcl::ONIGrabber (oni_file, repeat, ! triggered_capture));
                std::cout << "finished reading oni..." << std::endl;
            }
            else if (pc::parse_argument (argc, argv, "-pcd", pcd_dir) > 0)
            {
                float fps_pcd = 15.0f;
                pc::parse_argument (argc, argv, "-pcd_fps", fps_pcd);

                vector<string> pcd_files = getPcdFilesInDir(pcd_dir);

                // Sort the read files by name
                sort (pcd_files.begin (), pcd_files.end ());
                capture.reset (new pcl::PCDGrabber<pcl::PointXYZ> (pcd_files, fps_pcd, false));
            }
            else if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
            {
                //init data source latter
                pc::parse_argument (argc, argv, "-match_file", match_file);
            }
            else
            {
                capture.reset( new pcl::OpenNIGrabber() );

                //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224932.oni", true, ! triggered_capture) );
                //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/reg20111229-180846.oni, true, ! triggered_capture) );
                //triggered_capture = true; capture.reset( new pcl::ONIGrabber("/media/Main/onis/20111013-224932.oni", true, ! triggered_capture) );
                //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224551.oni", true, ! triggered_capture) );
                //triggered_capture = true; capture.reset( new pcl::ONIGrabber("d:/onis/20111013-224719.oni", true, ! triggered_capture) );
            }
        }
        catch (const pcl::PCLException& /*e*/) { return cout << "Can't open depth source" << endl, -1; }

        float volume_size = 3.f;
        pc::parse_argument (argc, argv, "-volume_size", volume_size);

        int icp = 1, visualization = 0;
        pc::parse_argument (argc, argv, "--icp", icp);
        pc::parse_argument (argc, argv, "--viz", visualization);
        std::cout << "visualisation: " << (visualization ? "yes" : "no") << std::endl;

        std::string outFileName = "cloud";
        pc::parse_argument (argc, argv, "-out", outFileName );

        KinFuApp app (*capture, volume_size, icp, visualization);

        if (pc::parse_argument (argc, argv, "-eval", eval_folder) > 0)
            app.toggleEvaluationMode(eval_folder, match_file);

        if (pc::find_switch (argc, argv, "--current-cloud") || pc::find_switch (argc, argv, "-cc"))
            app.initCurrentFrameView ();

        if (pc::find_switch (argc, argv, "--save-views") || pc::find_switch (argc, argv, "-sv"))
            app.image_view_.accumulate_views_ = true;  //will cause bad alloc after some time

        if (pc::find_switch (argc, argv, "--registration") || pc::find_switch (argc, argv, "-r"))
            app.initRegistration();

        if (pc::find_switch (argc, argv, "--integrate-colors") || pc::find_switch (argc, argv, "-ic"))
            app.toggleColorIntegration();

        if (pc::find_switch (argc, argv, "--scale-truncation") || pc::find_switch (argc, argv, "-st"))
            app.enableTruncationScaling();

        if (pc::find_switch (argc, argv, "--dump-poses") || pc::find_switch (argc, argv, "-dp"))
            app.dump_poses_ = true;

        app.scan_ = true;
        app.scan_volume_ = true;

        // executing
        try { app.startMainLoop (triggered_capture); }
        catch (const pcl::PCLException& /*e*/) { cout << "PCLException" << endl; }
        catch (const std::bad_alloc& /*e*/) { cout << "Bad alloc" << endl; }
        catch (const std::exception& /*e*/) { cout << "Exception" << endl; }

        std::cout << "writing..." << std::endl;
        app.writeCloud ( nsKinFuApp::PLY, outFileName );
        app.writeMesh ( nsKinFuApp::MESH_PLY, outFileName );
        //app.writeCloud ( nsKinFuApp::PCD_BIN, outFileName );
        app.saveTSDFVolume( outFileName );

#ifdef HAVE_OPENCV
        for (size_t t = 0; t < app.image_view_.views_.size (); ++t)
        {
            if (t == 0)
            {
                cout << "Saving depth map of first view." << endl;
                cv::imwrite ("./depthmap_1stview.png", app.image_view_.views_[0]);
                cout << "Saving sequence of (" << app.image_view_.views_.size () << ") views." << endl;
            }
            char buf[4096];
            sprintf (buf, "./%06d.png", (int)t);
            cv::imwrite (buf, app.image_view_.views_[t]);
            printf ("writing: %s\n", buf);
        }
#endif

        return 0;
    }

} // ns am
