/****************************************************************************
*                                                                           *
*  OpenNI 1.x Alpha                                                         *
*  Copyright (C) 2011 PrimeSense Ltd.                                       *
*                                                                           *
*  This file is part of OpenNI.                                             *
*                                                                           *
*  OpenNI is free software: you can redistribute it and/or modify           *
*  it under the terms of the GNU Lesser General Public License as published *
*  by the Free Software Foundation, either version 3 of the License, or     *
*  (at your option) any later version.                                      *
*                                                                           *
*  OpenNI is distributed in the hope that it will be useful,                *
*  but WITHOUT ANY WARRANTY; without even the implied warranty of           *
*  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the             *
*  GNU Lesser General Public License for more details.                      *
*                                                                           *
*  You should have received a copy of the GNU Lesser General Public License *
*  along with OpenNI. If not, see <http://www.gnu.org/licenses/>.           *
*                                                                           *
****************************************************************************/
//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------

#include "MaCvImageBroadcaster.h"
#include "BilateralFiltering.h"
#include "HomographyCalculator.h"
#include "prism_camera_parameters.h"

#include "io/Recorder.h"
#include "io/CvImageDumper.h"

#include "util/MaUtil.h"

#include <XnCppWrapper.h>
#include <XnOS.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include <iostream>
#include <math.h>

using namespace xn;

//---------------------------------------------------------------------------
// Defines
//---------------------------------------------------------------------------
//#define SAMPLE_XML_PATH "../../Config/SamplesConfig.xml"
#define SAMPLE_XML_PATH "../KinectNodesConfig.xml"

//---------------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------------
Context g_context;
DepthGenerator g_depth;
ImageGenerator g_image;
IRGenerator g_ir;

int toImageSpace( cv::Mat const& dep16, cv::Mat &out, TCalibData calibData )
{
    TCalibData prismKinect;

    cv::Mat_<ushort>::const_iterator itEnd = dep16.end<ushort>();
    uint x_d = 0, y_d = 0;
    for ( cv::Mat_<ushort>::const_iterator it = dep16.begin<ushort>(); it != itEnd; it++ )
    {
        // to 3D
        cv::Vec3d P3D;
        P3D[0] = (x_d - calibData.dep_intr.cx()) * (*it) / calibData.dep_intr.fx();
        P3D[1] = (y_d - calibData.dep_intr.cy()) * (*it) / calibData.dep_intr.fy();
        P3D[2] = (*it);

        // move to RGB space
        cv::Vec3d P3Dp;
        for ( uchar roww = 0; roww < 3; ++roww )
        {
            P3Dp[roww] =   calibData.R.at<double>(roww,0) * P3D[0]
                         + calibData.R.at<double>(roww,1) * P3D[1]
                         + calibData.R.at<double>(roww,2) * P3D[2]
                         - calibData.T.at<double>(roww);
        }

        cv::Vec2d P2D_rgb;
        P2D_rgb[0] = (P3Dp[0] * calibData.rgb_intr.fx() / P3Dp[2]) + calibData.dep_intr.cx();
        P2D_rgb[1] = (P3Dp[1] * calibData.dep_intr.fy() / P3Dp[2]) + calibData.dep_intr.cy();

        if ( out.at<ushort>( P2D_rgb[1], P2D_rgb[0] ) < *it )
            out.at<ushort>( P2D_rgb[1], P2D_rgb[0] ) = *it;

        // iterate coords
        if ( ++x_d == static_cast<uint>(dep16.cols) )
        {
            x_d = 0;
            ++y_d;
        }
    }

    return EXIT_SUCCESS;
}

// reverse
int toDepSpace( cv::Mat const& dep16, cv::Mat &out )
{
    out = cv::Mat::zeros( dep16.rows, dep16.cols, dep16.type() );

    assert( dep16.type() == CV_16UC1 );

    // http://labs.manctl.com/rgbdemo/index.php/Documentation/KinectCalibrationTheory
    double fx_rgb =  5.2921508098293293e+02;
    double fy_rgb =  5.2556393630057437e+02;
    double cx_rgb =  3.2894272028759258e+02;
    double cy_rgb =  2.6748068171871557e+02;
    double k1_rgb =  2.6451622333009589e-01;
    double k2_rgb = -8.3990749424620825e-01;
    double p1_rgb = -1.9922302173693159e-03;
    double p2_rgb =  1.4371995932897616e-03;
    double k3_rgb =  9.1192465078713847e-01;

    double fx_d   =  5.9421434211923247e+02;
    double fy_d   =  5.9104053696870778e+02;
    double cx_d   =  3.3930780975300314e+02;
    double cy_d   =  2.4273913761751615e+02;
    double k1_d   = -2.6386489753128833e-01;
    double k2_d   =  9.9966832163729757e-01;
    double p1_d   = -7.6275862143610667e-04;
    double p2_d   =  5.0350940090814270e-03;
    double k3_d   = -1.3053628089976321e+00;

    double r_data[9] =  {  9.9984628826577793e-01,  1.2635359098409581e-03, -1.7487233004436643e-02,
                          -1.4779096108364480e-03,  9.9992385683542895e-01, -1.2251380107679535e-02,
                           1.7470421412464927e-02,  1.2275341476520762e-02,  9.9977202419716948e-01 };
    cv::Mat R( 3, 3, CV_64FC1, r_data );

    double t_data[3] =
    {
        1.9985242312092553e-02,
        -7.4423738761617583e-04,
        -1.0916736334336222e-02
    };
    cv::Mat T( 3, 1, CV_64FC1, t_data );
    cv::Mat RInv = R.inv();

    cv::Mat_<ushort>::const_iterator itEnd = dep16.end<ushort>();
    uint x_d = 0, y_d = 0;
    for ( cv::Mat_<ushort>::const_iterator it = dep16.begin<ushort>(); it != itEnd; it++ )
    {
        // to 3D
        cv::Vec3d P3D;
        P3D[0] = (x_d - cx_d) * (*it) / fx_d;
        P3D[1] = (y_d - cy_d) * (*it) / fy_d;
        P3D[2] = (*it);

        // move to RGB space
        cv::Vec3d P3Dp;
        for ( uchar roww = 0; roww < 3; ++roww )
        {
            P3Dp[roww] =   RInv.at<double>(roww,0) * P3D[0]
                         + RInv.at<double>(roww,1) * P3D[1]
                         + RInv.at<double>(roww,2) * P3D[2]
                         - T.at<double>(roww);
        }

        cv::Vec2d P2D_rgb;
        P2D_rgb[0] = (P3Dp[0] * fx_rgb / P3Dp[2]) + cx_rgb;
        P2D_rgb[1] = (P3Dp[1] * fy_rgb / P3Dp[2]) + cy_rgb;

        if ( out.at<ushort>( P2D_rgb[1], P2D_rgb[0] ) < *it )
            out.at<ushort>( P2D_rgb[1], P2D_rgb[0] ) = *it;

        // iterate coords
        if ( ++x_d == static_cast<uint>(dep16.cols) )
        {
            x_d = 0;
            ++y_d;
        }
    }

    return EXIT_SUCCESS;
}

int toImageSpace( cv::Mat const& dep16, cv::Mat &out )
{
    /*
     *Focal Length:          fc = [ 1049.13200   1052.32643 ] � [ 4.27512   4.22598 ]
    Principal point:       cc = [ 614.13049   567.61659 ] � [ 4.45672   4.75006 ]
    Skew:             alpha_c = [ 0.00075 ] � [ 0.00108  ]   => angle of pixel axes = 89.95697 � 0.06205 degrees
    Distortion:            kc = [ 0.22526   -0.66357   0.00406   -0.00090  0.59980 ] � [ 0.01546   0.07547   0.00190   0.00168  0.11433 ]
    Pixel error:          err = [ 0.68181   0.61335 ]
*/
    out = cv::Mat::zeros( dep16.rows, dep16.cols, dep16.type() );

    assert( dep16.type() == CV_16UC1 );

    // http://labs.manctl.com/rgbdemo/index.php/Documentation/KinectCalibrationTheory
    double fx_rgb = 5.2921508098293293e+02;
    double fy_rgb = 5.2556393630057437e+02;
    double cx_rgb = 3.2894272028759258e+02;
    double cy_rgb = 2.6748068171871557e+02;
    double k1_rgb = 2.6451622333009589e-01;
    double k2_rgb = -8.3990749424620825e-01;
    double p1_rgb = -1.9922302173693159e-03;
    double p2_rgb = 1.4371995932897616e-03;
    double k3_rgb = 9.1192465078713847e-01;

    double fx_d = 5.9421434211923247e+02;
    double fy_d = 5.9104053696870778e+02;
    double cx_d = 3.3930780975300314e+02;
    double cy_d = 2.4273913761751615e+02;
    double k1_d = -2.6386489753128833e-01;
    double k2_d = 9.9966832163729757e-01;
    double p1_d = -7.6275862143610667e-04;
    double p2_d = 5.0350940090814270e-03;
    double k3_d = -1.3053628089976321e+00;

    double r_data[9] =  { 9.9984628826577793e-01, 1.2635359098409581e-03,
                          -1.7487233004436643e-02, -1.4779096108364480e-03,
                          9.9992385683542895e-01, -1.2251380107679535e-02,
                          1.7470421412464927e-02, 1.2275341476520762e-02,
                          9.9977202419716948e-01 };
    cv::Mat R( 3, 3, CV_64FC1, r_data );

    double t_data[3] = {
        1.9985242312092553e-02, -7.4423738761617583e-04,
        -1.0916736334336222e-02 };
    cv::Mat T( 3, 1, CV_64FC1, t_data );

    cv::Mat_<ushort>::const_iterator itEnd = dep16.end<ushort>();
    uint x_d = 0, y_d = 0;
    for ( cv::Mat_<ushort>::const_iterator it = dep16.begin<ushort>(); it != itEnd; it++ )
    {
        // to 3D
        cv::Vec3d P3D;
        P3D[0] = (x_d - cx_d) * (*it) / fx_d;
        P3D[1] = (y_d - cy_d) * (*it) / fy_d;
        P3D[2] = (*it);

        // move to RGB space
        cv::Vec3d P3Dp;
        for ( uchar roww = 0; roww < 3; ++roww )
        {
            P3Dp[roww] =   R.at<double>(roww,0) * P3D[0]
                         + R.at<double>(roww,1) * P3D[1]
                         + R.at<double>(roww,2) * P3D[2]
                         + T.at<double>(roww);
        }

        cv::Vec2d P2D_rgb;
        P2D_rgb[0] = (P3Dp[0] * fx_rgb / P3Dp[2]) + cx_rgb;
        P2D_rgb[1] = (P3Dp[1] * fy_rgb / P3Dp[2]) + cy_rgb;

        if ( out.at<ushort>( P2D_rgb[1], P2D_rgb[0] ) < *it )
            out.at<ushort>( P2D_rgb[1], P2D_rgb[0] ) = *it;

        // iterate coords
        if ( ++x_d == static_cast<uint>(dep16.cols) )
        {
            x_d = 0;
            ++y_d;
        }
    }

    return EXIT_SUCCESS;
}

int doMapping( /*  IN: */ cv::Mat const& dep16, cv::Mat const& img8,
               /* OUT: */ TMatDict &out_mats,
               /* FLG: */ bool doShow = true )
{
    static const double ratio = 255.0 / 2047.0;
    //static am::BilateralFiltering bf( 3.0, 3.0 );

    // check input
    assert( dep16.type() == CV_16UC1 );

    /// simple overlay (depth16 -> image)
    cv::Mat dep16Large, overlay = img8.clone();
    {
        // upscale depth
        cv::resize( dep16, dep16Large, overlay.size(), 0, 0, cv::INTER_NEAREST );

        // overlay on RGB
        uint y = 0, x = 0;
        cv::Mat_<ushort>::const_iterator itEnd = dep16Large.end<ushort>();
        for ( cv::Mat_<ushort>::const_iterator it = dep16Large.begin<ushort>(); it != itEnd; it++ )
        {
            // read
            uchar dVal = dep16Large.at<ushort>( y, x ) * ratio;
            if ( dVal )
            {
                //overlay.at<cv::Vec3b>( y, x ) = (cv::Vec3b){ dVal, dVal, dVal };
                overlay.at<cv::Vec3b>( y, x ) = util::blend( overlay.at<cv::Vec3b>( y, x ), dVal, 0.5 );
            }

            // iterate coords
            if ( ++x == static_cast<uint>(dep16Large.cols) )
            {
                x = 0;
                ++y;
            }
        }

        // output
        out_mats["overlay"] = overlay.clone();
    }

    /// registered small (dep16 -> map -> image)
    cv::Mat dep16Mapped, overlayMapped;
    {
        // map dep16
        toImageSpace( dep16, dep16Mapped );
        // downscale RGB
        cv::resize( img8, overlayMapped, dep16Mapped.size(), 0, 0, cv::INTER_LANCZOS4 );

        // overlay on RGB
        uint y = 0, x = 0;
        cv::Mat_<ushort>::const_iterator itEnd = dep16Mapped.end<ushort>();
        for ( cv::Mat_<ushort>::const_iterator it = dep16Mapped.begin<ushort>(); it != itEnd; it++ )
        {
            // read
            uchar dVal = dep16Mapped.at<ushort>( y, x ) * ratio;
            if ( dVal )
            {
                overlayMapped.at<cv::Vec3b>( y, x ) = util::blend( overlayMapped.at<cv::Vec3b>( y, x ), dVal, 0.6 );
            }

            // iterate coords
            if ( ++x == static_cast<uint>(dep16Mapped.cols) )
            {
                x = 0;
                ++y;
            }
        }

        // output
        //out_mats["dep16Mapped"] = dep16Mapped.clone();
        out_mats["overlayMapped"] = overlayMapped.clone();
    }

    /// registered large (dep16 -> map -> resize -> image)
    cv::Mat dep16MappedLarge, overlayMappedLarge = img8.clone();
    {
        // downscale RGB
        cv::resize( dep16Mapped, dep16MappedLarge, img8.size(), 0, 0, cv::INTER_NEAREST );

        // overlay on RGB
        cv::Mat_<ushort>::const_iterator itEnd = dep16MappedLarge.end<ushort>();
        uint y = 0, x = 0;
        for ( cv::Mat_<ushort>::const_iterator it = dep16MappedLarge.begin<ushort>(); it != itEnd; it++ )
        {
            // read
            uchar dVal = dep16MappedLarge.at<ushort>( y, x ) * ratio;
            if ( dVal )
            {
                overlayMappedLarge.at<cv::Vec3b>( y, x ) = util::blend( overlayMappedLarge.at<cv::Vec3b>( y, x ), dVal, 0.8f );
            }

            // iterate coords
            if ( ++x == static_cast<uint>(dep16MappedLarge.cols) )
            {
                x = 0;
                ++y;
            }
        }

        // output
        //out_mats["overlayMappedLarge"] = overlayMappedLarge.clone();
        out_mats["dep16MappedLarge"] = dep16MappedLarge.clone();
    }

    // IMSHOW
    if ( doShow )
    {
        for ( auto it = out_mats.begin(); it != out_mats.end(); ++it )
        {
            cv::imshow( it->first, it->second );
        }
    }

    // return
    return EXIT_SUCCESS;
}

void getIR( Context &context, IRGenerator &irGenerator, cv::Mat &irImage )
{
    context.WaitOneUpdateAll( irGenerator );
    XnIRPixel const* irMap = irGenerator.GetIRMap();
    unsigned int max = 0;
    for ( int i = 0; i < 640*480; ++i )
    {
        if (irMap[i] > max )
        {
            max = irMap[i];
        }
    }
    std::cout << "irmax: " << max << std::endl;
    /*for ( int i = 0; i < 640*480; ++i )
    {
        tempIR[i] = (int)( (double)irMap[i] / max*65535 );
    }
    cvSetData( irImage, (void*)tempIR, irImage->widthStep);
    cvSaveImage(filename, irImage);*/
    xn::IRMetaData irMD;
    cv::Mat cvIr16;
    irGenerator.GetMetaData( irMD );
    cvIr16.create( irMD.FullYRes(), irMD.FullXRes(), CV_16UC1 );

    // COPY
    {
        const XnIRPixel *pIrPixels = irMD.Data();
        //cvSetData( cvIr16, pIrPixels, cvIr16.step );
        //cvIr16 = cv::Mat( irMD.FullYRes(), irMD.FullXRes(), CV_16UC1, pIrPixels );
        int offset = 0;
        for ( XnUInt y = 0; y < irMD.YRes(); ++y, offset += irMD.XRes() )
            memcpy( cvIr16.data + cvIr16.step * y, pIrPixels + offset, irMD.XRes() * sizeof(XnIRPixel) );
    }

    cv::convertScaleAbs( cvIr16, irImage, 255.0/(double)max );
}

/**
 * @brief combineIRandRgb
 * @param ir8 typed uchar
 * @param rgb8 typed uchar3
 * @param size desired output size
 * @param out place for output
 */
void combineIRandRgb( cv::Mat &ir8, cv::Mat &rgb8, cv::Size size, cv::Mat &out, float alpha = .5f)
{
    CV_Assert( ir8.type() == CV_8UC1 );
    CV_Assert( rgb8.type() == CV_8UC3 );

    // downscale RGB
    cv::Mat tmp1, tmp2, *pIr = nullptr, *pRgb = nullptr;
    if ( ir8.size() != size )
    {
        tmp1.create( size.height, size.width, CV_8UC1 );
        pIr = &tmp1;
        cv::resize( ir8, *pIr, size, 0, 0, cv::INTER_NEAREST );
    }
    else
    {
        pIr = &ir8;
    }
    if ( rgb8.size() != size )
    {
        tmp2.create( size, CV_8UC3 );
        pRgb = &tmp2;
        cv::resize( rgb8, *pRgb, size, 0, 0, cv::INTER_NEAREST );
    }
    else
    {
        pRgb = &rgb8;
    }
    out.create( size, CV_8UC3 );

    // overlay on RGB
    cv::Mat_<uchar>::const_iterator itEnd = pIr->end<uchar>();
    uint y = 0, x = 0;
    for ( cv::Mat_<uchar>::const_iterator it = pIr->begin<uchar>(); it != itEnd; it++ )
    {
        // read
        uchar dVal = pIr->at<uchar>( y, x );
        if ( dVal )
        {
            out.at<cv::Vec3b>( y, x ) = util::blend( dVal, pRgb->at<cv::Vec3b>( y, x ), alpha );
        }

        // iterate coords
        if ( ++x == static_cast<uint>(pIr->cols) )
        {
            x = 0;
            ++y;
        }
    }
}

int playGenerators( Context &context, DepthGenerator& depthGenerator, ImageGenerator &imageGenerator, IRGenerator &irGenerator )
{
    XnStatus nRetVal = XN_STATUS_OK;

    // Calibration data
    TCalibData prismKinect;
    initPrismCamera( prismKinect );

    // declare CV
    cv::Mat dep8, dep16, rgb8, ir8;

    char c = 0;
    while ( (!xnOSWasKeyboardHit()) && (c != 27) )
    {
        // read DEPTH
        nRetVal = context.WaitOneUpdateAll( depthGenerator );
        if (nRetVal != XN_STATUS_OK)
        {
            printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
            continue;
        }

        if ( depthGenerator.IsGenerating() )
            util::nextDepthToMats( depthGenerator, &dep8, &dep16 );
        if ( imageGenerator.IsGenerating() )
            util::nextImageAsMat ( imageGenerator, &rgb8 );
        if ( irGenerator.IsGenerating() )
        {
            std::cout << "fetching ir..." << std::endl;
            getIR( context, irGenerator, ir8 );
            std::cout << "switching to rgb..." << std::endl;
            irGenerator.StopGenerating();
            imageGenerator.StartGenerating();
            std::cout << "fetching rgb..." << std::endl;
            context.WaitOneUpdateAll( imageGenerator );
            util::nextImageAsMat ( imageGenerator, &rgb8 );
            std::cout << "switching back to ir..." << std::endl;
            imageGenerator.StopGenerating();
            irGenerator.StartGenerating();
            std::cout << "finished..." << (irGenerator.IsGenerating() ? "ir is on" : "ir is off") << std::endl;
        }

        // Distribute
        TMatDict filtered_mats;
        /*if ( depthGenerator.IsGenerating() && imageGenerator.IsGenerating() )
            doMapping( dep16, img8, filtered_mats );*/

        /*if ( !ir8.empty() )
        {
            imshow("ir8", ir8 );
        }
        if ( !rgb8.empty() )
        {
            imshow("img8", rgb8 );
        }
        if ( !dep8.empty() )
        {
            imshow("dep8", dep8 );
        }*/

        cv::Mat irAndRgb;
        if ( !ir8.empty() && !rgb8.empty() )
        {
            combineIRandRgb( ir8, rgb8, rgb8.size(), irAndRgb );
            imshow( "irAndRgb", irAndRgb );

            am::CvImageDumper::Instance().dump( dep8    , "dep8"    , false  );
            am::CvImageDumper::Instance().dump( rgb8    , "img8"    , false  );
            am::CvImageDumper::Instance().dump( ir8     ,  "ir8"    , false  );
            am::CvImageDumper::Instance().dump( irAndRgb, "irAndRgb", true   );
        }

        c = cv::waitKey( 500 );
        switch ( c )
        {
            case 32:
                {
                    for ( auto it = filtered_mats.begin(); it != filtered_mats.end(); ++it )
                    {
                        am::CvImageDumper::Instance().dump( it->second, it->first, false );
                    }
                    am::CvImageDumper::Instance().dump( dep8, "dep8", false );
                    //am::CvImageDumper::Instance().dump( dep16, "dep16", false );
                    am::CvImageDumper::Instance().dump( rgb8, "img8", false  );
                    am::CvImageDumper::Instance().dump( ir8, "ir8", false  );
                    am::CvImageDumper::Instance().dump( irAndRgb, "irAndRgb", true );
                }
                break;
            case 's':
                {
                    std::cout << "switching...";
                    if ( irGenerator.IsGenerating() )
                    {
                        irGenerator.StopGenerating();
                        imageGenerator.StartGenerating();
                    }
                    else
                    {
                        imageGenerator.StopGenerating();
                        irGenerator.StartGenerating();
                    }
                    std::cout << "imageGenerator is " << (imageGenerator.IsGenerating() ? "ON" : "off")
                              << " irGenerator is " << (irGenerator.IsGenerating() ? "ON" : "off")
                              << std::endl;
                }
                break;
        }
    }
}

int doFiltering()
{
    std::string path = "/home/bontius/workspace/CppProjects/SuperRes-NI-1-5/out/imgs_20130607_1525/";
    TMatDict dict;
    dict["overlayMappedLarge"] = cv::imread( path + "overlayMappedLarge_00000003.png", -1 );
    dict["img8"] = cv::imread( path + "img8_00000003.png", -1 );
    dict["dep8"] = cv::imread( path + "dep8_00000003.png", -1 );
    dict["dep16"] = cv::imread( path + "dep16_00000003.png", -1 );
    dict["dep16MappedLarge"] = cv::imread( path + "dep16MappedLarge_00000003.png", -1 );
    imshow( "dep8", dict["dep8"] );
    imshow( "dep16", dict["dep16"] );

    double sigma1 = 5.0;
    double sigma2 = 0.4;

    static am::BilateralFiltering bf(sigma1, sigma2);
    cv::Mat cross16;
    bf.runFilter<ushort,uchar>( dict["dep16MappedLarge"], dict["img8"], cross16 );
    imshow("cross16",cross16);

    char fname[256];
    sprintf( fname, "cross16_%f_%f.png", sigma1, sigma2 );
    cv::imwrite( fname, cross16 );
    cv::Mat cross8;
    cv::convertScaleAbs( cross16, cross8, 255.0 / 2047.0 );
    sprintf( fname, "cross8_%f_%f.png", sigma1, sigma2 );
    cv::imwrite( fname, cross8 );

    cv::waitKey(0);
}

// predecl
int mainYet(int argc, char* argv[]);

int main(int argc, char* argv[])
{
    //mainYet( argc, argv );
    //return 0;

    //doFiltering();
    //return EXIT_SUCCESS;

    // CONFIG
    enum PlayMode { RECORD, PLAY, KINECT };
    PlayMode playMode = KINECT;

    // INPUT
    char* ONI_PATH = "recording_push.oni";
    if ( argc > 1 )
    {
        ONI_PATH = argv[1];
    }

    // VAR
    XnStatus rc;
    EnumerationErrors errors;
    xn::Player player;
    ScriptNode g_scriptNode;

    // dumping
    am::CvImageDumper::Instance().setOutputPath( "out/imgs" );

    // INIT
    switch ( playMode )
    {
        case PlayMode::RECORD:
            {
                am::Recorder rtest( ONI_PATH, SAMPLE_XML_PATH );
                rtest.setSamplePath( SAMPLE_XML_PATH );
                rtest.setAltViewpoint( false );

                return rtest.run( false );

                break;
            }

        case PlayMode::PLAY:
            {
                rc = g_context.Init();
                CHECK_RC( rc, "Init" );

                // open input file
                rc = g_context.OpenFileRecording( ONI_PATH, player );
                CHECK_RC( rc, "Open input file" );
                break;
            }

        case PlayMode::KINECT:
            {
                char path[1024];
                getcwd( path, 1024 );
                //std::cout << "initing " << path << "/" << SAMPLE_XML_PATH << std::endl;
                /*util::catFile( std::string(path) + "/" + SAMPLE_XML_PATH );*/
                //rc = g_context.InitFromXmlFile( SAMPLE_XML_PATH, &errors );
                break;
            }
    }

#define RGB_WIDTH 1280
#define RGB_HEIGHT 1024
#if 1
    /// init NODES
    XnMapOutputMode modeIR;
    modeIR.nFPS = 30;
    modeIR.nXRes = 640;
    modeIR.nYRes = 480;
    XnMapOutputMode modeVGA;
    modeVGA.nFPS = 15;
    modeVGA.nXRes = RGB_WIDTH;
    modeVGA.nYRes = RGB_HEIGHT;

    //context inizialization
    rc = g_context.Init();
    CHECK_RC(rc, "Initialize context");

    //depth node creation
    rc = g_depth.Create(g_context);
    CHECK_RC(rc, "Create depth generator");
    rc = g_depth.StartGenerating();
    CHECK_RC(rc, "Start generating Depth");

    //RGB node creation
    rc = g_image.Create(g_context);
    CHECK_RC(rc, "Create rgb generator");
    rc = g_image.SetMapOutputMode(modeVGA);
    CHECK_RC(rc, "Depth SetMapOutputMode XRes for 1280, YRes for 1024 and FPS for 15");
    //rc = g_image.StartGenerating();
    //CHECK_RC(rc, "Start generating RGB");

    //IR node creation
    rc = g_ir.Create(g_context);
    CHECK_RC(rc, "Create ir generator");
    rc = g_ir.SetMapOutputMode(modeIR);
    CHECK_RC(rc, "IR SetMapOutputMode XRes for 640, YRes for 480 and FPS for 30");
    rc = g_ir.StartGenerating();
    CHECK_RC(rc, "Start generating IR");
#else

    /// init NODES
    {
        rc = g_context.FindExistingNode(XN_NODE_TYPE_DEPTH, g_depth);
        if (rc != XN_STATUS_OK)
        {
            printf( "No depth node exists! %s\n", xnGetStatusString(rc) );
            //return 1;
        }

        rc = g_context.FindExistingNode(XN_NODE_TYPE_IMAGE, g_image);
        if (rc != XN_STATUS_OK)
        {
            printf( "No image node exists! %s\n", xnGetStatusString(rc) );
            //return 1;
        }

        rc = g_context.FindExistingNode(XN_NODE_TYPE_IR, g_ir);
        if (rc != XN_STATUS_OK)
        {
            printf( "No ir node exists! %s\n", xnGetStatusString(rc) );
            //return 1;
        }
    }

    // REGISTRATION
    //if ( 0 || playMode != PLAY )
    {
        if ( g_depth )
        {
            XnBool isSupported = g_depth.IsCapabilitySupported("AlternativeViewPoint");
            if ( isSupported == TRUE )
            {
                std::cout << "resetting viewpoint" << std::endl;
                XnStatus res = g_depth.GetAlternativeViewPointCap().ResetViewPoint();
                //XnStatus res = g_depth.GetAlternativeViewPointCap().SetViewPoint( g_image );
                if ( XN_STATUS_OK != res )
                {
                    printf("Getting and setting AlternativeViewPoint failed: %s\n", xnGetStatusString(res));
                }
            }}
    }
#endif

    /// RUN
    playGenerators( g_context, g_depth, g_image, g_ir );

    return EXIT_SUCCESS;
}
