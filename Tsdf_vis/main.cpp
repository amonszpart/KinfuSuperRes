#include "tsdf_viewer.h"

#include "BilateralFilterCuda.hpp"
#include "YangFilteringWrapper.h"

#include "my_screenshot_manager.h"

#include "MaUtil.h"

#include <pcl/io/vtk_lib_io.h>
#include <pcl/console/parse.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <eigen3/Eigen/Dense>
#include <boost/filesystem.hpp>

#include <map>
#include <string>
#include <iostream>

// --in ~/rec/troll_recordings/short_prism_kinect_20130816_1206/short_20130816_1327_nomap
// --in /home/amonszpart/rec/testing/ram_20130818_1209_lf_200/cloud_mesh.ply
// --in ~/workspace/rec/testing/ram_20130818_1209_lf_200/cloud_mesh.ply

// try to get ZBuffer image to the right scale
template <typename T>
void processZBuffer( std::vector<T> zBuffer, int w, int h, std::map<std::string,cv::Mat> mats )
{
    // zBuf to Mat
    cv::Mat zBufMat( h, w, CV_32FC1 );
    for ( int y = 0; y < zBufMat.rows; ++y )
    {
        for ( int x = 0; x < zBufMat.cols; ++x )
        {
            zBufMat.at<float>(y,x) = 1.f - zBuffer[ y * zBufMat.cols + x ];
        }
    }

    // minmax zBufMat
    double minv, maxv;
    cv::minMaxLoc( zBufMat, &minv, &maxv );
    std::cout << "maxv: " << maxv << " minv: " << minv << std::endl;

    // minmax large_dep16
    {
        double minVal, maxVal;
        cv::minMaxIdx( mats["large_dep16"], &minVal, &maxVal );
        std::cout << "minVal(large_dep16): " << minVal << ", "
                  << "maxVal(large_dep16): " << maxVal << std::endl;
    }

    // upscale
    cv::Mat tmp;
    cv::subtract( zBufMat, minv, tmp, cv::Mat(), CV_32FC1 );
    cv::divide( tmp, (maxv - minv) / 10001.f, zBufMat, CV_32FC1 );

    // show
    cv::imshow( "zBufMat", zBufMat / 10001.f );
    cv::imwrite( "zBufMat.png", zBufMat );

    cv::Mat rgb8_960;
    cv::resize( mats["rgb8"], rgb8_960, mats["large_dep16"].size() );
    cv::Mat zBuf8;
    zBufMat.convertTo( zBuf8, CV_8UC1, 1.f / 10001.f );

    std::vector<cv::Mat> zBuf8Vec = { zBuf8, zBuf8, zBuf8 };
    cv::Mat zBuf8C3;
    cv::merge( zBuf8Vec, zBuf8C3 );
    std::cout << "merge ok" << std::endl;

    cv::Mat overlay;
    cv::addWeighted( zBuf8C3, 0.9f, rgb8_960, 0.1f, 1.0, overlay );
    cv::imshow( "overlay", overlay );
}

// global state
struct MyPlayer
{
        bool exit;      // has to exit
        bool changed;   // has to redraw

        void *weak_cloud_viewer_ptr;

        MyPlayer()
            : exit( false ), changed( false )
        {}

        Eigen::Affine3f      & Pose()       { changed = true; return pose; }
        Eigen::Affine3f const& Pose() const {                 return pose; }

    protected:
        Eigen::Affine3f pose;

} g_myPlayer;

// all viewers keyboard callback
void keyboard_callback( const pcl::visualization::KeyboardEvent &e, void *cookie )
{
    MyPlayer* pMyPlayer = reinterpret_cast<MyPlayer*>( cookie );

    int key = e.getKeyCode ();

    if ( e.keyUp () )
    {
        switch ( key )
        {
            case 27:
                pMyPlayer->exit = true;
                break;
            case 82:
            case 'a':
                pMyPlayer->Pose().translation().x() -= 0.1f;
                break;
            case 'd':
            case 83:
                pMyPlayer->Pose().translation().x() += 0.1f;
                break;
            case 's':
            case 84:
                pMyPlayer->Pose().translation().y() -= 0.1f;
                break;
            case 'w':
            case 81:
                pMyPlayer->Pose().translation().y() += 0.1f;
                break;
            case 'e':
                pMyPlayer->Pose().translation().z() += 0.1f;
                break;
            case 'c':
                pMyPlayer->Pose().translation().z() -= 0.1f;
                break;

            default:
                break;
        }
    }
    std::cout << (int)key << std::endl;
}

// cloudviewer mouse callback
void mouse_callback (const pcl::visualization::MouseEvent& mouse_event, void* cookie)
{
    // player pointer
    MyPlayer* pMyPlayer = reinterpret_cast<MyPlayer*>( cookie );

    // left button release
    if ( mouse_event.getType()   == pcl::visualization::MouseEvent::MouseButtonRelease &&
         mouse_event.getButton() == pcl::visualization::MouseEvent::LeftButton            )
    {
        // debug
        std::cout << g_myPlayer.Pose().linear() << g_myPlayer.Pose().translation() << std::endl;

        // read
        Eigen::Affine3f tmp_pose = reinterpret_cast<pcl::visualization::PCLVisualizer*>(pMyPlayer->weak_cloud_viewer_ptr)->getViewerPose();

        // modify
        //tmp_pose.linear() = tmp_pose.linear().inverse();

        // write
        g_myPlayer.Pose() = tmp_pose; // reinterpret_cast<pcl::visualization::PCLVisualizer*>(pMyPlayer->weak_cloud_viewer_ptr)->getViewerPose();

        // debug
        std::cout << g_myPlayer.Pose().linear() << g_myPlayer.Pose().translation() << std::endl;
    }
}

// CLI usage
void printUsage()
{
    std::cout << "Usage:\n\tTSDFVis --in cloud.dat\n" << std::endl;
    std::cout << "\tYang usage: --yangd dir --dep depName --img imgName --iter yangIterationCount" << std::endl;
}

// main
int main( int argc, char** argv )
{
    //// YANG /////////////////////////////////////////////////////////////////////////////////////////////////////////
    {
        bool canDoYang = true;

        // input dir
        std::string yangDir;
        canDoYang &= pcl::console::parse_argument (argc, argv, "--yangd", yangDir) >= 0;
        if ( canDoYang )
        {
            // depth
            std::string depName;
            canDoYang &= pcl::console::parse_argument (argc, argv, "--dep", depName) >= 0;

            // image
            std::string imgName;
            canDoYang &= pcl::console::parse_argument (argc, argv, "--img", imgName) >= 0;

            // iterations
            int iterations = 3;
            canDoYang &= pcl::console::parse_argument (argc, argv, "--iter", iterations);
            if ( iterations <= 0 ) iterations = 3;
            std::cout << "Running for " << iterations << std::endl;

            // error check
            if ( !canDoYang )
            {
                std::cerr << "yang usage: --yangd dir --dep depName --img imgName" << std::endl;
                return EXIT_FAILURE;
            }

            // run
            runYang( yangDir + "/" + depName, yangDir + "/" + imgName, iterations );
        }
        // else TSDF or PLY
    }

    //// TSDF or PLY //////////////////////////////////////////////////////////////////////////////////////////////////

    std::map<std::string, cv::Mat> mats;

    // parse input
    std::string inputFilePath;
    if (pcl::console::parse_argument (argc, argv, "--in", inputFilePath) < 0 )
    {
        printUsage();
        return 1;
    }

    // flag yes, if PLY input
    bool ply_no_tsdf = false;
    if ( boost::filesystem::extension(inputFilePath) == ".ply")
    {
        std::cout << "ext: " << boost::filesystem::extension( inputFilePath ) << std::endl;
        ply_no_tsdf = true;
    }

    const int img_id = 50;

    // read RGB
    boost::filesystem::path rgb8_path = boost::filesystem::path(inputFilePath).parent_path()
                                        / std::string("poses")
                                        / (boost::lexical_cast<std::string> (img_id) + std::string(".png"));
    cv::Mat rgb8 = cv::imread( rgb8_path.c_str(), -1 );

    // read DEPTH
    boost::filesystem::path dep_path = boost::filesystem::path(inputFilePath).parent_path()
                                       / std::string("poses")
                                       / (std::string("d") + boost::lexical_cast<std::string> (img_id) + std::string(".png"));
    cv::Mat dep16 = cv::imread( dep_path.c_str(), -1 );
    cv::Mat large_dep16;
    cv::resize( dep16, large_dep16, dep16.size() * 2 );

    // debug
    cv::imshow( "rgb8", rgb8 );
    cv::imshow( "large_dep16", large_dep16 );
    cv::waitKey( 100 );

    // read poses
    std::map<int,Eigen::Affine3f> poses;
    {
        boost::filesystem::path poses_path = boost::filesystem::path(inputFilePath).parent_path()
                                             / std::string("poses")
                                             / "poses.txt";

        am::MyScreenshotManager::readPoses( poses_path.string(), poses );
    }

    // Load TSDF
    am::TSDFViewer *tsdfViewer = new am::TSDFViewer( ply_no_tsdf );
    if ( ply_no_tsdf )
    {
        // init pointer
        tsdfViewer->MeshPtr() = pcl::PolygonMesh::Ptr( new pcl::PolygonMesh() );
        // load mesh
        pcl::io::loadPolygonFile( inputFilePath, *tsdfViewer->MeshPtr() );
    }
    else
    {
        // load tsdf
        tsdfViewer->loadTsdfFromFile( inputFilePath, true );
        // register keyboard callbacks
        tsdfViewer->getRayViewer()->registerKeyboardCallback (keyboard_callback, (void*)&g_myPlayer);
        tsdfViewer->getDepthViewer()->registerKeyboardCallback (keyboard_callback, (void*)&g_myPlayer);
        // check, if mesh is valid
        tsdfViewer->dumpMesh();
    }

    // mouse callback, prepare myplayer global state
    tsdfViewer->getCloudViewer()->registerMouseCallback( mouse_callback, (void*)&g_myPlayer );
    g_myPlayer.weak_cloud_viewer_ptr = tsdfViewer->getCloudViewer().get();

    // apply pose
    g_myPlayer.Pose() = poses[ img_id ];

    // test intrinsics
    Eigen::Matrix3f intr_m3f;
    intr_m3f << 521.7401, 0       , 323.4402 * 2.f,
                0       , 522.1379, 258.1387 * 2.f,
                0       , 0       , 1             ;


    std::vector<float> zBuffer;
    int w, h;
    while ( (!g_myPlayer.exit) && (!tsdfViewer->getCloudViewer()->wasStopped()) )
    {
        if ( g_myPlayer.changed )
        {
            if ( !ply_no_tsdf )
            {
                // raycast
                tsdfViewer->showGeneratedRayImage( tsdfViewer->kinfuVolume_ptr_, g_myPlayer.Pose() );
                // tsdf depth
                tsdfViewer->showGeneratedDepth   ( tsdfViewer->kinfuVolume_ptr_, g_myPlayer.Pose() );

                // point cloud
                //tsdfViewer->toCloud( myPlayer.Pose(), tsdfViewer->CloudPtr() );
                //tsdfViewer->showCloud(  myPlayer.Pose(), tsdfViewer->CloudPtr() );
                // range image
                //tsdfViewer->renderRangeImage( tsdfViewer->CloudPtr(), myPlayer.Pose() );
            }

            // show mesh
            tsdfViewer->showMesh(  g_myPlayer.Pose(), tsdfViewer->MeshPtr() );
            // set pose
            tsdfViewer->setViewerPose( *tsdfViewer->getCloudViewer(), g_myPlayer.Pose() );
            //tsdfViewer->getCloudViewer()->setCameraParameters( intr_m3f, myPlayer.Pose().matrix() );
            // dump zbuffer
            tsdfViewer->vtkMagic( zBuffer, w, h );

            g_myPlayer.changed = false;
        }
        // update
        tsdfViewer->spinOnce(10);

        // exit after one iteration
        //myPlayer.exit = true;
    }

    // process Z buffer
    mats["large_dep16"] = large_dep16;
    mats["rgb8"] = rgb8;
    processZBuffer( zBuffer, w, h, mats );

    // wait for exit
    char c = 0;
    while ( (c = cv::waitKey()) != 27 ) ;

    // dump latest tsdf depth
    if ( !ply_no_tsdf )
    {
        // get depth
        auto vshort = tsdfViewer->getLatestDepth();
        cv::Mat dep( 480, 640, CV_16UC1, vshort.data() );
        util::writePNG( "dep224.png", dep );
    }

    // cleanup
    {
        if ( tsdfViewer ) { delete tsdfViewer; tsdfViewer = NULL; }
        std::cout << "Tsdf_vis finished ok" << std::endl;
    }

} // end main

#if 0
    Eigen::Affine3f pose;
    pose.linear() <<  0.999154,  -0.0404336, -0.00822448,
            0.0344457,    0.927101,   -0.373241,
            0.0227151,    0.372638,    0.927706;
    pose.translation() << 1.63002,
            1.46289,
            0.227635;

    // 224
    //Eigen::Affine3f pose;
    myPlayer.Pose().translation() << 1.67395, 1.69805, -0.337846;

    myPlayer.Pose().linear() <<
                     0.954062,  0.102966, -0.281364,
                    -0.16198,  0.967283, -0.195268,
                    0.252052,  0.231873,  0.939525;
#endif
