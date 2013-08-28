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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/vtk_lib_io.h>
#include <eigen3/Eigen/Dense>


int main( int argc, char **argv )
{
    am::TriangleRenderer rendererInstance;

    // projection
    Eigen::Matrix3f intrinsics;
//    intrinsics << 521.7401 * 2.f, 0       , 323.4402 * 2.f,
//            0             , 522.1379 * 2.f, 258.1387 * 2.f,
//            0             , 0             , 1             ;
    intrinsics << 521.7401 * 2.f / 1280.f * 32.f, 0.f       , 323.4402 * 2.f / 1280.f * 32.f,
            0.f             , 522.1379 * 2.f / 960.f * 24.f, 258.1387 * 2.f / 960.f * 24.f,
            0.f             , 0.f             , 1.f             ;

    // view
    Eigen::Affine3f pose; pose.linear()
            << 1.f, 0.f, 0.f,
               0.f, 1.f, 0.f,
               0.f, 0.f, 1.f;
    pose.translation() << 1.5f, 1.5f, -0.3f;

    std::vector<cv::Mat> depths, indices;
    cv::Mat depthFC1, indices32UC1;
    rendererInstance.renderDepthAndIndices( depths, indices, 32, 24, intrinsics, pose, "/home/bontius/workspace/rec/testing/cloud_mesh.ply", 10001.f );

    depthFC1     = depths[0];  // depths
    indices32UC1 = indices[0]; // vertex ids

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

