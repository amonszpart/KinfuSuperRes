#include "./tools/kinfu_app.h"

#include "BilateralFilterCuda.hpp"

#include "../util/MaUtil.h"
#include <iostream>

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace am
{
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

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    // -oni ~/rec/troll_recordings/keyboard_imgs_20130701_1440/recording_push.oni -out ~/rec/testing/keyboard_cross_nomap -ic --viz 1 -r
    // optionally: --dump-poses
    int
    mainKinfuApp (int argc, char* argv[])
    {
        if ( pc::find_switch (argc, argv, "--help") ||
             pc::find_switch (argc, argv, "-h"    )    )
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
            if (pc::parse_argument (argc, argv, "-dev" , openni_device) > 0)
            {
                capture.reset (new pcl::OpenNIGrabber (openni_device));
            }
            else if (pc::parse_argument (argc, argv, "-oni" , oni_file     ) > 0)
            {
                triggered_capture = true;
                bool repeat = false; // Only run ONI file once
                std::cout << "trying to read oni: " << oni_file << "...";
                capture.reset (new pcl::ONIGrabber (oni_file, repeat, ! triggered_capture));
                std::cout << "YES, will use oni as input source..." << std::endl;
            }
            else if (pc::parse_argument (argc, argv, "-pcd" , pcd_dir      ) > 0)
            {
                float fps_pcd = 15.0f;
                pc::parse_argument (argc, argv, "-pcd_fps", fps_pcd);

                vector<string> pcd_files = getPcdFilesInDir(pcd_dir);

                // Sort the read files by name
                sort (pcd_files.begin (), pcd_files.end ());
                capture.reset (new pcl::PCDGrabber<pcl::PointXYZ> (pcd_files, fps_pcd, false));
            }
            else if (pc::parse_argument (argc, argv, "-eval", eval_folder  ) > 0)
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
        pc::parse_argument ( argc, argv, "-volume_size", volume_size );

        int icp = 1, visualization = 0;
        pc::parse_argument ( argc, argv, "--icp", icp );
        pc::parse_argument ( argc, argv, "--viz", visualization );
        std::cout << "Visualisation: " << (visualization ? "yes" : "no") << std::endl;

        std::string outFileName = "cloud";
        pc::parse_argument ( argc, argv, "-out", outFileName );

        KinFuApp app ( *capture, volume_size, icp, visualization );

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

# if 1
        // save computations to files
        {
            std::string path = util::outputDirectoryNameWithTimestamp( outFileName ) + "/";
            xnOSCreateDirectory( path.c_str() );
            std::cout << "writing to " << path << "..." << std::endl;
            app.writeCloud    ( nsKinFuApp::PLY     , path+"cloud" );
            app.writeMesh     ( nsKinFuApp::MESH_PLY, path+"cloud" );
            app.saveTSDFVolume( path + "cloud" );
            app.writeCloud ( nsKinFuApp::PCD_BIN, path+"cloud" );
        }
#endif
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
}
