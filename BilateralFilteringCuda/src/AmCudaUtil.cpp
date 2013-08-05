#include "AmCudaUtil.h"

#include "builtin_types.h"
#include <opencv2/core/core.hpp>
#include <iostream>

void cv2Continuous8UC4(cv::Mat const& img, unsigned*& hImg, unsigned width, unsigned height, float alpha )
{
    if ( hImg )
        free( hImg );
    hImg = (unsigned int *) malloc( img.cols * img.rows * sizeof(unsigned) );
    uchar4** pHImage = reinterpret_cast<uchar4**>( &hImg );

    for ( int yi = img.rows; yi; --yi )
        for ( int x = 0; x < img.cols; ++x )
        {
            int y = yi-1;
            if ( img.type() == CV_16UC1 )
            {
                ushort p = img.at<ushort>( y, x );
                (*pHImage)[y * img.cols + x].x = (uchar)( (float)p * alpha );
                (*pHImage)[y * img.cols + x].y = (uchar)( (float)p * alpha );
                (*pHImage)[y * img.cols + x].z = (uchar)( (float)p * alpha );
                (*pHImage)[y * img.cols + x].w = 0;
            }
            else if ( img.type() == CV_8UC1 )
            {
                uchar p = img.at<uchar>( y, x );
                (*pHImage)[y * img.cols + x].x = p;
                (*pHImage)[y * img.cols + x].y = p;
                (*pHImage)[y * img.cols + x].z = p;
                (*pHImage)[y * img.cols + x].w = 0;
            }
            else if ( img.type() == CV_8UC3 )
            {
                cv::Vec3b p = img.at<cv::Vec3b>( y, x );
                (*pHImage)[y * img.cols + x].x = p[2];
                (*pHImage)[y * img.cols + x].y = p[1];
                (*pHImage)[y * img.cols + x].z = p[0];
                (*pHImage)[y * img.cols + x].w = 0;
            }
            else
            {
                std::cerr << "unknown image format..." << std::endl;
                exit(1);
            }
        }

    width  = img.cols;
    height = img.rows;
}
