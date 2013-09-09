#include "UpScaling.h"

#include "TriangleRenderer.h"
#include "PolyMeshViewer.h"
#include "YangFilteringWrapper.h"
#include "MeshRayCaster.h"
#include "AmPclUtil.h"
#include "MaUtil.h"
#include "AMUtil2.h"

#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <pcl/console/parse.h>

#include <opencv2/highgui/highgui.hpp>

#define SUBDIV_ITERATIONS (0)

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
        std::string outDir = ::util::outputDirectoryNameWithTimestamp( polygonPath.parent_path().string() + "/enhanced" );
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

        pcl::PolygonMesh::Ptr subdivMeshPtr( new pcl::PolygonMesh );
#if (SUBDIV_ITERATIONS > 0)
        // SUBDIV
        std::cout << "UpScaling::run(): subdividing mesh...";
        pcl::PolygonMesh::Ptr subdivMeshPtr( new pcl::PolygonMesh );
        MeshRayCaster::subdivideMesh( *subdivMeshPtr, mesh, SUBDIV_ITERATIONS );
        //*subdivMeshPtr = *mesh;
        std::cout << "OK..." << std::endl;
#else
        *subdivMeshPtr = *mesh;
#endif

        MeshRayCaster meshRayCaster( intrinsics_ );
        cv::Mat zBufMat; // depth map
#if 1
        std::cout << "UpScaling::run(): showing mesh using trianglerenderer...";

        //am::TriangleRenderer triangleRenderer;
        std::vector<cv::Mat> depths, indices;
        am::TriangleRenderer::Instance().renderDepthAndIndices( /* out: */ depths, indices,
                                                                /*  in: */ cols, rows, intrinsics_, pose, mesh,
                                                                /* depths[0] scale: */ 1.f );
        depths[0].copyTo( zBufMat );
        float zMax = 10.1f;

#elif 1


        // DepthMap
        std::cout << "UpScaling::run(): showing mesh...";
        PolyMeshViewer polyMeshViewer( intrinsics_, cols, rows );
        polyMeshViewer.initViewer( "UpScaling inputMesh" );
        polyMeshViewer.showMesh( subdivMeshPtr, pose );
        std::cout << "OK..." << std::endl;

        std::cout << "mesh: mesh->cloud.size: " << mesh->cloud.width << "x" <<  mesh->cloud.height << std::endl;
        std::cout << "subdivMeshPtr->cloud.size: " << subdivMeshPtr->cloud.width << "x" <<  subdivMeshPtr->cloud.height << std::endl;

        std::cout << "UpScaling::run(): fetching Z buffer...";

        am::util::pcl::fetchViewerZBuffer( zBufMat, polyMeshViewer.VisualizerPtr() );
        std::cout << "OK..." << std::endl;
#else
        cv::Mat zBufMat(rows,cols,CV_16UC1);
        meshRayCaster.run( zBufMat, subdivMeshPtr, pose, 0 );
#endif
        // DepthMap (dump)
        {
            cv::Mat zBufMat8;
            zBufMat.convertTo( zBufMat8, CV_8UC1, 255.f / zMax );
            cv::imwrite( outDir + "/zBufMat8" + ((img_id > -1) ? ("_" + boost::lexical_cast<std::string>(img_id)) : "") + ".png", zBufMat8 );
        }

        // BLEND (dump)
        cv::Mat blended;
        {
            am::util::blend( blended, zBufMat, zMax, rgb8, 255.f );
            cv::imshow( "blended", blended);
            cv::waitKey(10);
            cv::imwrite( outDir + "/blended"  + ((img_id > -1) ? ("_" + boost::lexical_cast<std::string>(img_id)) : "") + ".png", blended );
        }

        // YANG
        cv::Mat filtered;
        {
            std::cout << "UpScaling::run(): Yang...";

            YangFilteringRunParams params;
            params.spatial_sigma = 1.2;
            params.range_sigma = 0.1;
            params.kernel_range = 5;
            params.yang_iterations = 10;
            params.yang_iterations = 3;
            params.L = 40;

            pcl::console::parse_argument( argc, argv, "--spatial_sigma"   , params.spatial_sigma    );
            pcl::console::parse_argument( argc, argv, "--range_sigma"     , params.range_sigma      );
            pcl::console::parse_argument( argc, argv, "--kernel_range"    , params.kernel_range     );
            pcl::console::parse_argument( argc, argv, "--cross_iterations", params.cross_iterations );
            pcl::console::parse_argument( argc, argv, "--iter"            , params.yang_iterations  );
            runYangCleaned( filtered, zBufMat, rgb8, params );
            std::cout << "OK..." << std::endl;
        }

        // Filtered (dump)
        cv::Mat filtered8;
        {
            filtered.convertTo( filtered8, CV_8UC1, 255.f/zMax);
            cv::imwrite( outDir + "/zBufMat8"  + ((img_id > -1) ? ("_" + boost::lexical_cast<std::string>(img_id)) : "") + "_filtered.png", filtered8 );
        }

        if ( (filtered.type() != CV_32FC1) || (zBufMat.type() != CV_32FC1) )
        {
            std::cerr << "went back to short too soon..." << std::endl;
        }

        // Diff
        cv::Mat diff( filtered.rows, filtered.cols, CV_16UC1 );
        {
            for ( int y = 0; y < filtered.rows; ++y )
                for ( int x = 0; x < filtered.cols; ++x )
                {
                    diff.at<ushort>( y, x ) = 10001U + (ushort)round((filtered.at<float>(y,x) - zBufMat.at<float>(y,x)) * 10001.f/zMax);
                }
            cv::imwrite( outDir + "/zBufMat_diff16UC1"  + ((img_id > -1) ? ("_" + boost::lexical_cast<std::string>(img_id)) : "") + ".png", diff );
        }

        // Enhance
        pcl::PolygonMesh::Ptr enhancedMeshPtr( new pcl::PolygonMesh );
        {
            std::cout << "UpScaling::run(): enhance mesh...";
#if (SUBDIV_ITERATIONS > 0)
            std::cout << "subdivMeshPtr->cloud.size: " << subdivMeshPtr->cloud.width << "x" <<  subdivMeshPtr->cloud.height << std::endl;
            std::cout << "polyMeshViewer.MeshPtr(): mesh->cloud.size: " << subdivMeshPtr->cloud.width << "x" <<  subdivMeshPtr->cloud.height << std::endl;
            meshRayCaster.enhanceMesh( enhancedMeshPtr, filtered, subdivMeshPtr, pose, 3.f / 640.f );
#else
            //meshRayCaster.enhanceMesh2( enhancedMeshPtr, filtered, mesh, pose, depths, indices );
            {
                double minVal, maxVal;
                cv::minMaxIdx( zBufMat, &minVal, &maxVal );
                std::cout << "minVal(zBufMat): " << minVal << ", "
                          << "maxVal(zBufMat): " << maxVal << std::endl;
            }
            {
                double minVal, maxVal;
                cv::minMaxIdx( filtered, &minVal, &maxVal );
                std::cout << "minVal(filtered): " << minVal << ", "
                          << "maxVal(filtered): " << maxVal << std::endl;
            }
            meshRayCaster.enhanceMesh2( enhancedMeshPtr, filtered, mesh, pose, depths, indices );
#endif
            std::cout << "OK..." << std::endl;
        }

        /*PolyMeshViewer polyMeshViewer2( intrinsics_, cols, rows );
        polyMeshViewer2.initViewer( "UpScaling outputMesh" );
        polyMeshViewer2.showMesh( enhancedMeshPtr, pose );*/

        // SAVE
        {
            std::cout << "UpScaling::run(): save mesh...";
            std::cout << outName;
            pcl::io::savePLYFile( outName, *enhancedMeshPtr );
#if (SUBDIV_ITERATIONS > 0)
            pcl::io::savePLYFile( outSubdivName, *subdivMeshPtr );
#endif
            std::cout << "...OK..." << std::endl;
        }

        //polyMeshViewer2.VisualizerPtr()->spin();
        system( ("meshlab " + outName + " &").c_str() );

        std::cout << "UpScaling::run(): FINISHED..." << std::endl;
    }

    int
    UpScaling::depthEdgeBlend( /* out: */ cv::Mat &blended, /*  in: */ cv::Mat const& depth, cv::Mat const& rgb, float dmax )
    {
        if ( depth.empty() ) { std::cerr << "depth is EMPTY" << std::endl; return EXIT_FAILURE; }
        if ( rgb  .empty() ) { std::cerr << "rgb is EMPTY"   << std::endl; return EXIT_FAILURE; }

        if ( depth.type() != CV_32FC1 ) { std::cerr << "testDepthEdge(): depth not float..." << std::endl; return EXIT_FAILURE; }

        cv::Mat edgesX, edgesY, edges;
        cv::Sobel( depth, edgesX, CV_32FC1, 1, 0 );
        cv::Sobel( depth, edgesY, CV_32FC1, 0, 1 );
        cv::addWeighted( edgesX, 127.5f/10001.f, edgesY, 127.5f/10001.f, 0, edges );

        //cv::Mat rgb8_960;
        //cv::resize( rgb, large_rgb8, depth.size(), 0, 0, CV_INTER_NN );
        //ViewPointMapperCuda::undistortRgb( rgb8_960, rgb, am::viewpoint_mapping::INTR_RGB_1280_1024, am::viewpoint_mapping::INTR_RGB_1280_1024 );

        am::util::cv::blend( blended, edges, 1.f, rgb );
    }

} // end ns am
