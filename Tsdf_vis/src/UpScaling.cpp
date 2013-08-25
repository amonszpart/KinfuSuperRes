#include "UpScaling.h"

#include "PolyMeshViewer.h"
#include "YangFilteringWrapper.h"
#include "MeshRayCaster.h"
#include "AmPclUtil.h"
#include "MaUtil.h"

#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/console/parse.h>

#include <opencv2/highgui/highgui.hpp>



namespace am
{

    UpScaling::UpScaling()
        : ProcessorWithIntrinsics()
    {
    }

    UpScaling::UpScaling( Eigen::Matrix3f intrinsics )
        : ProcessorWithIntrinsics( intrinsics )
    {
    }

    void
    UpScaling::run( std::string const& sPolygonPath, Eigen::Affine3f const& pose, cv::Mat const& rgb8, int img_id, int p_cols, int p_rows, int argc, char** argv )
    {
        int cols = ( p_cols > -1 ) ? p_cols : rgb8.cols;
        int rows = ( p_rows > -1 ) ? p_rows : rgb8.rows;

        std::cout << "UpScaling::run(): starting at res: " << cols << "x" << rows << std::endl;

        boost::filesystem::path polygonPath( sPolygonPath );
        std::string outDir = ::util::outputDirectoryNameWithTimestamp( polygonPath.parent_path().string() + "enhanced" );
        boost::filesystem::create_directory( outDir );
        std::string outName = outDir
                              + "/" + polygonPath.stem().string()
                              + "_enhanced"
                              + ((img_id > -1) ? ("_" + boost::lexical_cast<std::string>(img_id)) : "")
                              + ".ply";
        std::string outSubdivName = outDir
                              + "/" + polygonPath.stem().string()
                              + "_subdiv"
                              + ((img_id > -1) ? ("_" + boost::lexical_cast<std::string>(img_id)) : "")
                              + ".ply";



        pcl::PolygonMesh::Ptr mesh( new pcl::PolygonMesh );
        pcl::io::loadPolygonFile( sPolygonPath, *mesh );

        std::cout << "UpScaling::run(): subdividing mesh...";
        pcl::PolygonMesh::Ptr subdivMeshPtr( new pcl::PolygonMesh );
        //MeshRayCaster::subdivideMesh( *subdivMeshPtr, mesh, 1 );
        *subdivMeshPtr = *mesh;
        std::cout << "OK..." << std::endl;

        MeshRayCaster meshRayCaster( intrinsics_ );
#if 0
        std::cout << "UpScaling::run(): showing mesh...";
        PolyMeshViewer polyMeshViewer( intrinsics_, cols, rows );
        polyMeshViewer.initViewer( "UpScaling inputMesh" );
        polyMeshViewer.showMesh( subdivMeshPtr, pose );
        std::cout << "OK..." << std::endl;

        std::cout << "mesh: mesh->cloud.size: " << mesh->cloud.width << "x" <<  mesh->cloud.height << std::endl;
        std::cout << "subdivMeshPtr->cloud.size: " << subdivMeshPtr->cloud.width << "x" <<  subdivMeshPtr->cloud.height << std::endl;

        std::cout << "UpScaling::run(): fetching Z buffer...";
        cv::Mat zBufMat;
        am::util::pcl::fetchViewerZBuffer( zBufMat, polyMeshViewer.VisualizerPtr() );
        std::cout << "OK..." << std::endl;
#else
        cv::Mat zBufMat(rows,cols,CV_16UC1);
        meshRayCaster.run( zBufMat, subdivMeshPtr, pose, 0 );
#endif

        cv::Mat blended;
        am::util::blend( blended, zBufMat, 10001.f, rgb8, 255.f );
        cv::imshow( "blended", blended);
        cv::waitKey(10);
        cv::imwrite("blended.png", blended );

        // debug
        cv::Mat zBufMat8;
        zBufMat.convertTo( zBufMat8, CV_8UC1, 255.f/ 10001.f );
        cv::imwrite( outDir + "/zBufMat8.png", zBufMat8 );

        std::cout << "UpScaling::run(): Yang...";
        cv::Mat filtered;
        YangFilteringRunParams params;
        params.spatial_sigma = 1.2;
        params.range_sigma = 0.1;
        params.kernel_range = 5;
        params.yang_iterations = 10;

        pcl::console::parse_argument( argc, argv, "--spatial_sigma"   , params.spatial_sigma    );
        pcl::console::parse_argument( argc, argv, "--range_sigma"     , params.range_sigma      );
        pcl::console::parse_argument( argc, argv, "--kernel_range"    , params.kernel_range     );
        pcl::console::parse_argument( argc, argv, "--cross_iterations", params.cross_iterations );
        pcl::console::parse_argument( argc, argv, "--iter"            , params.yang_iterations  );
//        params.spatial_sigma = 1.2;
//        params.range_sigma = 0.1;
//        params.kernel_range = 3;
//        params.yang_iterations = 1;
        runYangCleaned( filtered, zBufMat, rgb8, params );
        std::cout << "OK..." << std::endl;

        // debug
        cv::Mat filtered8;
        filtered.convertTo( filtered8, CV_8UC1, 255.f/10001.f);
        cv::imwrite( outDir + "/zBufMat8_filtered.png", filtered8 );

        std::cout << "UpScaling::run(): enhance mesh...";
        pcl::PolygonMesh::Ptr enhancedMeshPtr( new pcl::PolygonMesh );

        std::cout << "polyMeshViewer.MeshPtr(): mesh->cloud.size: " << subdivMeshPtr->cloud.width << "x" <<  subdivMeshPtr->cloud.height << std::endl;
        std::cout << "subdivMeshPtr->cloud.size: " << subdivMeshPtr->cloud.width << "x" <<  subdivMeshPtr->cloud.height << std::endl;
        meshRayCaster.enhanceMesh( enhancedMeshPtr, filtered, subdivMeshPtr, pose, 3.f / 640.f );
        std::cout << "OK..." << std::endl;

        std::cout << "UpScaling::run(): save mesh...";
        std::cout << outName;
        pcl::io::savePLYFile( outName, *enhancedMeshPtr );
        pcl::io::savePLYFile( outSubdivName, *subdivMeshPtr );
        std::cout << "...OK..." << std::endl;

        std::cout << "UpScaling::run(): FINISHED..." << std::endl;
    }

} // end ns am
