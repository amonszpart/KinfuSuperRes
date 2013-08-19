#include "tsdf_viewer.h"

#include "DepthViewer3D.h"

#include "BilateralFilterCuda.hpp"
#include "YangFilteringWrapper.h"

#include "my_screenshot_manager.h"

#include "MaUtil.h"

#include <pcl/ros/conversions.h>
#include <pcl/octree/octree.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_subdivision.h>
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
    std::cout << "\tYang usage: --yangd dir --dep depName --img imgName"
              << " [--spatial_sigma x]"
              << " [--range_sigma x]"
              << " [--kernel_range x]"
              << " [--cross_iterations x]"
              << " [--iter yangIterationCount]"
              << std::endl;
}

void octtreeMesh( pcl::PolygonMesh &mesh )
{
    float resolution = .05f;

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree( resolution );

    //Declaration and instantiation of a cloud pointer to be used for output
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_pcl(new pcl::PointCloud<pcl::PointXYZ>);

    //Input of the above cloud and the corresponding output of cloud_pcl
    pcl::fromPCLPointCloud2( mesh.cloud, *cloud_pcl );
    //pcl::fromROSMsg( mesh.cloud, *cloud_pcl );
    octree.setInputCloud( cloud_pcl );
    octree.addPointsFromInputCloud();

    // Neighbors within voxel search
    pcl::PointXYZ searchPoint;
    searchPoint.x = 0;
    searchPoint.y = 3;
    searchPoint.z = 1;

    std::vector<int> pointIdxVec;

    if ( octree.voxelSearch(searchPoint, pointIdxVec) )
    {
        std::cout << "Neighbors within voxel search at (" << searchPoint.x
                  << " " << searchPoint.y
                  << " " << searchPoint.z
                  << ")" << std::endl;


        for ( size_t i = 0; i < pointIdxVec.size (); ++i )
            std::cout << "    " << cloud_pcl->points[pointIdxVec[i]].x
                      << " " << cloud_pcl->points[pointIdxVec[i]].y
                      << " " << cloud_pcl->points[pointIdxVec[i]].z << std::endl;
    }

    std::cout << "PointIdxVec size:" << pointIdxVec.size() << std::endl;

    std::cout << "Octree Leaf Count Operation: " << octree.getLeafCount() << std::endl;
    std::cout << "Octree Branch Count Operation: " << octree.getBranchCount() << std::endl;
}

void subdivideMesh( pcl::PolygonMesh::ConstPtr mesh, pcl::PolygonMesh &output )
{
    pcl::MeshSubdivisionVTK msvtk;
    msvtk.setInputMesh( mesh );
    msvtk.setFilterType( pcl::MeshSubdivisionVTK::LINEAR );
    msvtk.process( output );
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

            if ( canDoYang )
            {
                if ( pcl::console::find_switch( argc, argv, "--brute-force") > 0 )
                {
                    bruteRun( yangDir + "/" + depName, yangDir + "/" + imgName );
                    return EXIT_SUCCESS;
                }
            }

            YangFilteringRunParams runParams;

            pcl::console::parse_argument( argc, argv, "--spatial_sigma"   , runParams.spatial_sigma    );
            pcl::console::parse_argument( argc, argv, "--range_sigma"     , runParams.range_sigma      );
            pcl::console::parse_argument( argc, argv, "--kernel_range"    , runParams.kernel_range     );
            pcl::console::parse_argument( argc, argv, "--cross_iterations", runParams.cross_iterations );
            pcl::console::parse_argument( argc, argv, "--iter"            , runParams.yang_iterations  );
            if ( runParams.yang_iterations <= 0 ) runParams.yang_iterations = 3;
            std::cout << "Running for " << runParams.yang_iterations << std::endl;


            // error check
            if ( !canDoYang )
            {
                printUsage();
                return EXIT_FAILURE;
            }

            // run
            //runYang( yangDir + "/" + depName, yangDir + "/" + imgName, runParams );
            return EXIT_SUCCESS;
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

    // read DEPTH
    cv::Mat dep16, large_dep16;
    {
        boost::filesystem::path dep_path = boost::filesystem::path(inputFilePath).parent_path()
                                           / std::string("poses")
                                           / (std::string("d") + boost::lexical_cast<std::string> (img_id) + std::string(".png"));
        dep16 = cv::imread( dep_path.c_str(), -1 );
        cv::resize( dep16, large_dep16, dep16.size() * 2 );
    }

    // read RGB
    cv::Mat rgb8, rgb8_960;
    {
        boost::filesystem::path rgb8_path = boost::filesystem::path(inputFilePath).parent_path()
                                            / std::string("poses")
                                            / (boost::lexical_cast<std::string> (img_id) + std::string(".png"));
        rgb8 = cv::imread( rgb8_path.c_str(), -1 );
        cv::resize( rgb8, rgb8_960, large_dep16.size() );
    }

    // debug
    {
        cv::imshow( "rgb8_960"   , rgb8_960    );
        cv::imshow( "large_dep16", large_dep16 );
        cv::waitKey( 100 );
    }

    // read poses
    std::map<int,Eigen::Affine3f> poses;
    {
        boost::filesystem::path poses_path = boost::filesystem::path(inputFilePath).parent_path()
                                             / std::string("poses")
                                             / "poses.txt";

        am::MyScreenshotManager::readPoses( poses_path.string(), poses );
    }

    // apply pose
    Eigen::Matrix3f intr_m3f;
    {
        g_myPlayer.Pose() = poses[ img_id ];

        // test intrinsics
        intr_m3f << 521.7401, 0       , 323.4402 * 2.f,
                    0       , 522.1379, 258.1387 * 2.f,
                    0       , 0       , 1             ;
    }

    // Load TSDF
    am::TSDFViewer *tsdfViewer = new am::TSDFViewer( ply_no_tsdf );
    {
        // mouse callback, prepare myplayer global state
        tsdfViewer->getCloudViewer()->registerMouseCallback( mouse_callback, (void*)&g_myPlayer );
        g_myPlayer.weak_cloud_viewer_ptr = tsdfViewer->getCloudViewer().get();

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
    }

    // octree
    {
        //if ( tsdfViewer->MeshPtr() ) octtreeMesh( *tsdfViewer->MeshPtr() );
    }

    // subdivision
    {
        pcl::PolygonMesh::Ptr subdividedMeshPtr = pcl::PolygonMesh::Ptr( new pcl::PolygonMesh() );
        std::cout << "Subdividing mesh...";
        if ( tsdfViewer->MeshPtr() ) subdivideMesh( tsdfViewer->MeshPtr(), *subdividedMeshPtr );
        std::cout << "OK" << std::endl;

        std::cout << "Copying subdivided mesh...";
        tsdfViewer->MeshPtr() = subdividedMeshPtr;
        std::cout << "OK" << std::endl;
    }

    // mats to 3D
    {
        DepthViewer3D depthViewer;
        depthViewer.showMats( large_dep16, rgb8_960, img_id, poses, intr_m3f );
#if 0
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr colorCloudPtr;
        matsTo3D<ushort>( large_dep16, rgb8_960, colorCloudPtr, intr_m3f );

        pcl::visualization::PCLVisualizer::Ptr pColorCloudViewer;
        pColorCloudViewer = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer("Color Cloud Viewer") );

        pColorCloudViewer->setBackgroundColor( 0, 0, 0 );
        pColorCloudViewer->addCoordinateSystem( 1.0 );
        pColorCloudViewer->initCameraParameters();
        pColorCloudViewer->setPosition( 0, 500 );
        pColorCloudViewer->setSize( 1280, 960 );
        pColorCloudViewer->setCameraClipDistances( 0.01, 10.01 );
        pColorCloudViewer->setShowFPS( false );

        Eigen::Vector4f centroid;
        pcl::compute3DCentroid( *colorCloudPtr, centroid );
        std::cout << "centroid: " << centroid << " " << centroid.cols() << " " << centroid.rows()
                  << centroid.block<3,1>(0,0).transpose() << std::endl;

        Eigen::Vector3f up = am::TSDFViewer::getViewerCameraUp( *pColorCloudViewer );
        //Eigen::Matrix3f r = Eigen::AngleAxisf( 0.25 * M_PI, up).toRotationMatrix();
        Eigen::Affine3f pose = poses[ img_id ] * Eigen::AngleAxisf( 0.5 * M_PI, (centroid.block<3,1>(0,0) + up).normalized() );
        //pose.linear() *= r;
        am::TSDFViewer::setViewerPose( *pColorCloudViewer, pose     );
        //am::TSDFViewer::setViewerFovy( *pColorCloudViewer, intr_m3f );
        pcl::visualization::PointCloudColorHandlerRGBField<pcl::PointXYZRGB> rgb( colorCloudPtr );
        pColorCloudViewer->addPointCloud<pcl::PointXYZRGB>( colorCloudPtr, rgb, "colorCloud" );

        pColorCloudViewer->spin();
#endif
    }

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
    {
        mats["large_dep16"] = large_dep16;
        mats["rgb8"]        = rgb8;
        processZBuffer( zBuffer, w, h, mats );
    }

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
