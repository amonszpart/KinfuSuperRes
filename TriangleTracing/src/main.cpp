#include <iostream>

using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "textfile.h"

#define M_PI       3.14159265358979323846

#include "mesh.h"
#include "TriangleRenderer.h"
#include "AMUtil2.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/vtk_lib_io.h>
#include <eigen3/Eigen/Dense>

#include <iomanip>


int main( int argc, char **argv )
{
    //am::TriangleRenderer rendererInstance;
    float w = 1280.f;
    float h = 960.f;

    // projection
    Eigen::Matrix3f intrinsics;
//    intrinsics << 521.7401 * 2.f, 0       , 323.4402 * 2.f,
//            0             , 522.1379 * 2.f, 258.1387 * 2.f,
//            0             , 0             , 1             ;
    intrinsics << 521.7401 * 2.f / 1280.f * w, 0.f       , 323.4402 * 2.f / 1280.f * w,
            0.f             , 522.1379 * 2.f / 960.f * h, 258.1387 * 2.f / 960.f * h,
            0.f             , 0.f             , 1.f             ;

    // view
    Eigen::Affine3f pose; pose.linear()
            <<  0.997822f,-0.0618835f,-0.0228414f,
               0.0633844f,0.995374f,0.072196f,
               0.018268f,-0.0734866f,0.997129f;
    pose.translation() << 1.51232, 1.49073, -0.211046;
    //pose.translation() << 1.51232, 2.49073, -0.211046;

    //0.997822,-0.0618835,-0.0228414,1.51232,
    //0.0633844,0.995374,0.072196,1.49073,
    //0.018268,-0.0734866,0.997129,-0.211046,0,0,0,1

    std::vector<cv::Mat> depths, indices;
    cv::Mat depthFC1, indices32UC1;
    am::TriangleRenderer::Instance().renderDepthAndIndices( depths, indices, w, h, intrinsics, pose,
                                                            "/home/bontius/workspace_local/long640_20130829_1525_200_400/cloud_mesh.ply", 1.f, /* showWindow: */ false );

    depthFC1     = depths[0];  // depths
    indices32UC1 = indices[0]; // vertex ids

    double minVal, maxVal;
    {
        cv::minMaxIdx( depthFC1, &minVal, &maxVal );
        std::cout << "minVal(depthFC1): " << std::setprecision(32) << minVal << ", "
                  << "maxVal(depthFC1): " << std::setprecision(32) << maxVal << std::endl;
    }

    am::util::savePFM( depthFC1, "kinfu.pfm" );
    cv::Mat tmp;
    depthFC1.convertTo( tmp, CV_8UC1, 255.f / maxVal );
    cv::imwrite( "kinfu8.png", tmp );

    cv::imshow( "depth", depthFC1 / maxVal );

    cv::waitKey();
    return 0;

    cv::imshow( "depth", depthFC1 / 5001.f );
    for ( int y = 0; y < indices32UC1.rows; ++y )
        for ( int x = 0; x < indices32UC1.cols; ++x )
        {
            std::cout << "("
                      << y << ","
                      << x << ","
                      << indices32UC1.at<unsigned>(y,x) << ","
                      <<   indices[1].at<unsigned>(y,x) << ","
                      << "  " << indices32UC1.at<unsigned>(y,x)/3
                      << ");";
        }
    std::cout << std::endl;

    cv::Mat indices2;
    indices32UC1.convertTo( indices2, CV_32FC1, 600000.f );
    cv::imshow( "indicesFC1", indices2 );
    {
        double minVal, maxVal;
        cv::minMaxIdx( indices2, &minVal, &maxVal );
        std::cout << "minVal(indices2): " << minVal << ", "
                  << "maxVal(indices2): " << maxVal << std::endl;
    }


    cv::Mat out;
    depthFC1.convertTo( out, CV_16UC1 );
    cv::imwrite( "distances.png", out );

    {
        double minVal, maxVal;
        cv::minMaxIdx( depthFC1, &minVal, &maxVal );
        std::cout << "minVal(depth): " << minVal << ", "
                  << "maxVal(depth): " << maxVal << std::endl;
    }

    cv::Mat indicesFC1( indices32UC1.rows, indices32UC1.cols, CV_32FC1 );
    for ( int y = 0; y < indicesFC1.rows; ++y )
    {
        for ( int x = 0; x < indicesFC1.cols; ++x )
        {
            //indicesFC1.at<float>( y, x ) = ((float)*((unsigned*)indices32UC1.ptr<uchar>(y, x * 4)));
            //indicesFC1.at<float>( y, x ) = ((float)*((unsigned*)indices32UC1.ptr<uchar>(y, x * 4)));
            indicesFC1.at<float>( y, x ) = indices32UC1.at<float>( y, x );
        }
    }

    {
        double minVal, maxVal;
        cv::minMaxIdx( indicesFC1, &minVal, &maxVal );
        std::cout << "minVal(indicesFC1): " << minVal << ", "
                  << "maxVal(indicesFC1): " << maxVal << std::endl;
    }

    cv::imshow( "indices", indicesFC1 / 530679.f );

    cv::Mat indices16UC1;
    indicesFC1.convertTo( indices16UC1, CV_16UC1 );
    cv::imwrite( "incidex.png", indices16UC1 );

    cv::waitKey();

    return(0);
}

