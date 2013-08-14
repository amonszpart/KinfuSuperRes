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

#include "BilateralFiltering.h"
#include "prism_camera_parameters.h"

#include "ViewPointMapperCuda.h"
#include "BilateralFilterCuda.hpp"
#include "MyThrustUtil.h"
#include "YangFiltering.h"

#include "io/Recorder.h"
#include "io/CvImageDumper.h"

#include "util/MaUtil.h"

#include <XnCppWrapper.h>
#include <XnOS.h>

#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "qx_constant_time_bilateral_filter_published.h"

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

//---------------------------------------------------------------------------
// Predeclarations
//---------------------------------------------------------------------------
int mainYet( int argc, char* argv[] );
class StopWatchInterface;
extern double bilateralFilterRGBA(uint *dDest,
                           int width, int height,
                           float e_d, int radius, int iterations,
                           StopWatchInterface *timer,
                           uint* dImage, uint* dTemp, uint pitch );

//---------------------------------------------------------------------------
// Functions
//---------------------------------------------------------------------------

void getIR( Context &context, IRGenerator &irGenerator, cv::Mat &irImage )
{
    //context.WaitOneUpdateAll( irGenerator );
    xn::IRMetaData irMD;
    cv::Mat cvIr16;
    irGenerator.GetMetaData( irMD );
    cvIr16.create( irMD.FullYRes(), irMD.FullXRes(), CV_16UC1 );

    unsigned int max = 0;
    // COPY
    {
        const XnIRPixel *pIrPixels = irMD.Data();
        //cvSetData( cvIr16, pIrPixels, cvIr16.step );
        //cvIr16 = cv::Mat( irMD.FullYRes(), irMD.FullXRes(), CV_16UC1, pIrPixels );
        int offset = 0;
        for ( XnUInt y = 0; y < irMD.YRes(); ++y, offset += irMD.XRes() )
        {
            memcpy( cvIr16.data + cvIr16.step * y, pIrPixels + offset, irMD.XRes() * sizeof(XnIRPixel) );

            for ( XnUInt x = 0; x < irMD.XRes(); ++x )
            {
                if ( pIrPixels[ offset + x ] > max )
                {
                    max = pIrPixels[ offset + x ];
                }
            }
        }
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

template <typename depT,typename imgT>
void overlay( cv::Mat const& dep, cv::Mat const& img, cv::Mat& out, depT maxDepth, imgT maxColor, float alpha = .5f )
{
    // init output
    out.create( img.rows, img.cols, img.type() );

    cv::Mat depClone;
    bool needsResize = ( (dep.size().width  != img.size().width ) ||
                         (dep.size().height != img.size().height)    );
    if ( needsResize ) cv::resize( dep, depClone, img.size(), 0, 0, cv::INTER_NEAREST );

    const cv::Mat *pDep = (needsResize) ? &depClone
                                        : &dep;

    // overlay on RGB
    uint y = 0, x = 0;
    auto itEnd = pDep->end<depT>();
    for ( auto it = pDep->begin<depT>(); it != itEnd; it++ )
    {
        // read
        depT dVal = pDep->at<depT>( y, x );
        if ( dVal != 0 )
        {
            //overlay.at<cv::Vec3b>( y, x ) = (cv::Vec3b){ dVal, dVal, dVal };
            out.at<imgT>( y, x ) = util::blend( img.at<imgT>( y, x ), dVal, alpha );
        }

        // iterate coords
        if ( ++x == static_cast<uint>(pDep->cols) )
        {
            x = 0;
            ++y;
        }
    }
}

struct Filtering
{
        static const int BORDER_TYPE = cv::BORDER_REPLICATE;

        static void guidedFilterSrc1Guidance1( const cv::Mat& src, const cv::Mat& joint, cv::Mat& dest, const int radius, const float eps )
        {
            if ( src.channels() != 1 && joint.channels() != 1 )
            {
                std::cout << "Please input gray scale image." << std::endl;
                return;
            }
            //some opt
            cv::Size ksize( 2 * radius + 1, 2 * radius + 1 );
            cv::Size imsize = src.size();
            const float e = eps * eps;

            cv::Mat fSrc;    src  .convertTo( fSrc  , CV_32F, 1.0/255 );
            cv::Mat fJoint;  joint.convertTo( fJoint, CV_32F, 1.0/255 );
            cv::Mat meanSrc  ( imsize, CV_32F ); // mean_p
            cv::Mat meanJoint( imsize, CV_32F ); // mean_I

            cv::boxFilter( fJoint, meanJoint, CV_32F, ksize, cv::Point(-1,-1), true, BORDER_TYPE ); // mJoint * K
            cv::boxFilter( fSrc  , meanSrc  , CV_32F, ksize, cv::Point(-1,-1), true, BORDER_TYPE ); // mSrc   * K

            cv::Mat x1( imsize, CV_32F ),
                    x2( imsize, CV_32F ),
                    x3( imsize, CV_32F );

            cv::multiply ( fJoint, fSrc, x1 );                                       // x1 * 1
            cv::boxFilter( x1, x3, CV_32F, ksize, cv::Point(-1,-1), true, BORDER_TYPE ); // x3 * K
            cv::multiply ( meanJoint, meanSrc, x1 );                                 // x1 = K * K
            x3   -= x1;                                                          // x1 div k -> x3 * k
            cv::multiply ( fJoint, fJoint, x1 );
            cv::boxFilter( x1, x2, CV_32F, ksize, cv::Point(-1,-1), true, BORDER_TYPE ); // x2 * K
            cv::multiply ( meanJoint, meanJoint, x1 ); // x1 = K * K
            fSrc = cv::Mat( x2 - x1 ) + e;
            cv::divide   ( x3, fSrc     , x3 );
            cv::multiply ( x3, meanJoint, x1 );
            x1   -= meanSrc;
            cv::boxFilter( x3, x2, CV_32F, ksize, cv::Point(-1,-1), true, BORDER_TYPE ); // x2 * k
            cv::boxFilter( x1, x3, CV_32F, ksize, cv::Point(-1,-1), true, BORDER_TYPE ); // x3 * k
            cv::multiply ( x2, fJoint, x1 );                                         // x1 * K
            cv::Mat x1_m_x3 = x1 - x3;
            x1_m_x3.convertTo( dest, src.type(), 255 );
        }

};

struct MyCVPlayer
{
        static void run()
        {
            cv::VideoCapture capture( CV_CAP_OPENNI );

            if( !capture.isOpened() )
            {
                std::cout << "Can not open a capture object." << std::endl;
            }
        }
};

//// TRACKBAR CALLBACKS
void on_contrast_alpha_trackbar( int, void* );
void on_contrast_beta_trackbar( int, void* );

// "crossFiltered8"
void on_cross_gaussian_delta_trackbar( int, void* );
void on_cross_eucledian_delta_trackbar( int, void* );
void on_cross_filter_range_trackbar( int, void* );
void on_cross_filter_iterations_trackbar( int, void* );


struct MyTrackbar
{
        int slider;
        int slider_max;
        float value;
        MyTrackbar( int slider, int slider_max, float value )
            : slider(slider), slider_max(slider_max), value(value) {}
};

struct MyPlayer
{
        bool showIR  = false;
        bool showRgb = false;
        bool showDep8 = false;
        bool showIrAndRgb = true;
        bool showDep16AndRgb = true;
        bool showOffset = false;
        bool altViewPoint = false;
        bool showGuided = false;

        int alpha_slider = 367;
        int alpha_slider_max = 500;
        int beta_slider = 5;
        int beta_slider_max = 255;
        float alpha = 1.f;
        float beta  = 0.f;

        // "crossFiltered8"
        const char *CROSS_WINDOW_NAME      = "crossFiltered8";
        MyTrackbar cross_gaussian_delta    = MyTrackbar( 100, 500, 1.f );
        MyTrackbar cross_eucledian_delta   = MyTrackbar( 100, 1000, .1f );
        MyTrackbar cross_filter_range      = MyTrackbar( 2, 50, 2 );
        MyTrackbar cross_filter_iterations = MyTrackbar( 1, 10, 1 );
        // "crossFiltered8"
        const char *BIL_WINDOW_NAME     = "bilFiltered8";
        MyTrackbar bil_gaussian_delta   = MyTrackbar( 100, 500, 1.f );
        MyTrackbar bil_eucledian_delta  = MyTrackbar( 100, 1000, .1f );
        MyTrackbar bil_filter_range     = MyTrackbar( 2, 50, 2 );

        int toggleAltViewpoint()
        {
            if ( g_depth )
            {
                if ( g_depth.IsCapabilitySupported("AlternativeViewPoint") == TRUE )
                {
                    if ( !altViewPoint ) // attempt enforce the toggled state
                    {
                        XnStatus res = g_depth.GetAlternativeViewPointCap().SetViewPoint( g_image );
                        if ( XN_STATUS_OK != res )
                        {
                            printf("Setting AlternativeViewPoint failed: %s\n", xnGetStatusString(res));
                            return res;
                        }
                    }
                    else
                    {
                        XnStatus res = g_depth.GetAlternativeViewPointCap().ResetViewPoint();
                        if ( XN_STATUS_OK != res )
                        {
                            printf("Reset AlternativeViewPoint failed: %s\n", xnGetStatusString(res));
                            return res;
                        }
                    }

                    // apply change, if succeeded
                    altViewPoint = !altViewPoint;
                    std::cout << "AltViewPoint is now: " << util::printBool( altViewPoint ) << std::endl;

                }
                else
                {
                    std::cerr << "AltViewpoint is not supported..." << std::endl;
                    return 1;
                }
            }
            else
            {
                std::cerr << "DepthGenerator is null..." << std::endl;
                return 1;
            }

            return 0;
        }

        int toggleIR()
        {
            if ( showIR = !showIR )
            {
                g_image.StopGenerating();
                g_ir.StartGenerating();
                cv::namedWindow( "ir8" );
                cv::createTrackbar( "alpha", "ir8", &alpha_slider, alpha_slider_max, on_contrast_alpha_trackbar );
                cv::createTrackbar( "beta", "ir8", &beta_slider, beta_slider_max, on_contrast_beta_trackbar );
            }
            else
            {
                g_ir.StopGenerating();
            }

        }

        int playGenerators( Context &context, DepthGenerator& depthGenerator, ImageGenerator &imageGenerator, IRGenerator &irGenerator )
        {
            XnStatus nRetVal = XN_STATUS_OK;

            // Calibration data
            //TCalibData prismKinect;
            //initPrismCamera( prismKinect );

            // declare CV
            cv::Mat dep8, dep16, rgb8, ir8, rgb8_1280;

            // init windows
            cv::namedWindow( CROSS_WINDOW_NAME );
            cv::createTrackbar( "cross_gaussian_delta (0..5.0)", CROSS_WINDOW_NAME, &cross_gaussian_delta.slider, cross_gaussian_delta.slider_max, on_cross_gaussian_delta_trackbar );
            cv::createTrackbar( "cross_eucledian_delta (0..1.0)", CROSS_WINDOW_NAME, &cross_eucledian_delta.slider, cross_eucledian_delta.slider_max, on_cross_eucledian_delta_trackbar );
            cv::createTrackbar( "cross_filter_range", CROSS_WINDOW_NAME, &cross_filter_range.slider, cross_filter_range.slider_max, on_cross_filter_range_trackbar );
            cv::createTrackbar( "cross_filter_iterations", CROSS_WINDOW_NAME, &cross_filter_iterations.slider, cross_filter_iterations.slider_max, on_cross_filter_iterations_trackbar );

            /*cv::namedWindow( CROSS_WINDOW_NAME );
            cv::createTrackbar( "cross_gaussian_delta", CROSS_WINDOW_NAME, &cross_gaussian_delta.slider, cross_gaussian_delta.slider_max, on_cross_gaussian_delta_trackbar );
            cv::createTrackbar( "cross_eucledian_delta", CROSS_WINDOW_NAME, &cross_eucledian_delta.slider, cross_eucledian_delta.slider_max, on_cross_eucledian_delta_trackbar );
            cv::createTrackbar( "cross_filter_range", CROSS_WINDOW_NAME, &cross_filter_range.slider, cross_filter_range.slider_max, on_cross_filter_range_trackbar );*/

            char c = 0;
            while ( (!xnOSWasKeyboardHit()) && (c != 27) )
            {
                // read DEPTH
                {
                    nRetVal = context.WaitOneUpdateAll( depthGenerator );
                    if (nRetVal != XN_STATUS_OK)
                    {
                        printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
                        continue;
                    }

                    if ( depthGenerator.IsGenerating() )
                        util::nextDepthToMats( depthGenerator, &dep8, &dep16 );
                }

                // read RGB
                if ( imageGenerator.IsGenerating() )
                {
                    util::nextImageAsMat ( imageGenerator, &rgb8 );
                    if ( rgb8.cols > dep8.cols )
                    {
                        rgb8.copyTo( rgb8_1280 );
                        cv::resize( rgb8_1280, rgb8, dep8.size(), 0, 0, cv::INTER_LANCZOS4 );
                    }


                }

                // read IR
                if ( irGenerator.IsGenerating() )
                {
                    std::cout << "fetching ir..." << std::endl;
                    getIR( context, irGenerator, ir8 );
                    std::cout << "fetched ir..." << std::endl;
                }

#if 1
                // read IR and RGB
                if ( showIrAndRgb && irGenerator.IsGenerating() )
                {
                    std::cout << "fetching ir...";
                    getIR( context, irGenerator, ir8 );

                    std::cout << "switching to rgb...";
                    irGenerator.StopGenerating();
                    imageGenerator.StartGenerating();
                    c = cv::waitKey(100);
                    context.WaitOneUpdateAll( imageGenerator );

                    std::cout << "fetching rgb...";
                    context.WaitOneUpdateAll( imageGenerator );

                    std::cout << "switching back to ir..." << std::endl;
                    imageGenerator.StopGenerating();
                    irGenerator.StartGenerating();
                    std::cout << "finished..." << (irGenerator.IsGenerating() ? "ir is on" : "ir is off") << std::endl;

                    util::nextImageAsMat ( imageGenerator, &rgb8 );
                }
#endif

                // Distribute
                TMatDict mats;

                // show Depth8
                if ( showDep8 && !dep8.empty() )
                {
                    imshow("dep8", dep8 );
                }

                // show RGB
                if ( showRgb && !rgb8.empty() )
                {
                    imshow("rgb8", rgb8 );
                }

                // show IR
                if ( showIR && !ir8.empty() )
                {
                    ir8.convertTo( ir8, ir8.type(), alpha, beta );
                    //cv::medianBlur( ir8, ir8, 3);
                    imshow("ir8", ir8 );
                }

                // mapping
                cv::Mat mapped16, mapped8;
                {
                    ViewPointMapperCuda::runViewpointMapping( dep16, mapped16 );
                    cv::imshow( "mapped16", mapped16 );
                    mapped16.convertTo( mapped8, CV_8UC1, 255.f / 10001.f );
                    cv::imshow( "mapped8", mapped8 );
                }

                // filtering
                //cv::Mat crossFiltered16;
                {
                    if ( !showGuided )
                    {
                        // crossFiltering
                        static BilateralFilterCuda<float> bfc;
                        bfc.setIterations( cross_filter_iterations.value );
                        bfc.runBilateralFiltering( mapped16, rgb8, mats["crossFiltered16"],
                                cross_gaussian_delta.value, cross_eucledian_delta.value, cross_filter_range.value );
                        mats["crossFiltered16"].convertTo( mats["crossFiltered8"], CV_8UC1, 255.f / 10001.f );
                        cv::imshow( CROSS_WINDOW_NAME, mats["crossFiltered8"] );
                    }
                    else
                    {
#if 0
                        // guided filtering
                        cv::Mat rgb8Gray;
                        // convert img
                        cv::cvtColor( rgb8, rgb8Gray, CV_RGB2GRAY );

                        Filtering::guidedFilterSrc1Guidance1( mapped8, rgb8Gray, mats["crossFiltered8"], 0.005, 8 );
                        mats["crossFiltered8"].convertTo( mats["crossFiltered16"], CV_16UC1, 10001.f / 255.f );
                        cv::imshow( CROSS_WINDOW_NAME, mats["crossFiltered8"] );
                        cv::imshow( "guided16", mats["crossFiltered16"] );

                        //cv::Mat fGuided255;
                        //mats["fGuided"].convertTo( fGuided255, CV_8UC1 );
                        //cv::imshow( "fGuided255", fGuided255 );

                        //cv::Mat x1;
                        //cv::addWeighted( rgb8Gray, .5, fGuided255, .5, .0, x1, CV_8UC1 );
                        //cv::imshow( "overlay(rgb,guided)", x1 );
#endif
                    }
                }


                // diff filtered
                cv::Mat fMapped16;
                {
                    cv::Mat fMapped16;
                    mapped16.convertTo( fMapped16, CV_32FC1, 1.f / 10001.f );

                    cv::Mat fFiltered;
                    mats["crossFiltered16"].convertTo( fFiltered, CV_32FC1, 1.f / 10001.f );

                    cv::Mat diff;
                    cv::absdiff( fMapped16, fFiltered, diff );
                    cv::imshow( "absdiff(crossFiltered16,mapped16)", diff );
                }

                // gradient filtered
                {
                    cv::Mat edgesX, edgesY, edgesMapped8, edgesCrossFiltered8;

                    // canny mapped8
                    cv::Canny( mapped8, edgesMapped8, .01f, .3f );
                    cv::imshow( "Canny(Mapped16)", edgesMapped8 );

                    // canny cross8
                    cv::Canny( mats["crossFiltered8"], edgesCrossFiltered8, .01f, .3f );
                    cv::imshow( "Canny(crossFiltered16)", edgesCrossFiltered8 );

                    // sobel
                    /*cv::Sobel( mats["crossFiltered8"], edgesX, mats["crossFiltered8"].type(), 1, 0 );
                    cv::Sobel( mats["crossFiltered8"], edgesY, mats["crossFiltered8"].type(), 0, 1 );
                    cv::addWeighted( edgesX, 0.5, edgesY, 0.5, 0, edges );*/

                    // diff
                    cv::Mat diff;
                    cv::absdiff( edgesMapped8, edgesCrossFiltered8, diff );
                    cv::imshow( "absdiff(edgesMapped8,edgesCrossFiltered8)", diff );
                }

                // guided filtering
                {
                    cv::Mat rgb8Gray, fGuided;
                    // convert img
                    cv::cvtColor( rgb8, rgb8Gray, CV_RGB2GRAY );

                    Filtering::guidedFilterSrc1Guidance1( mapped8, rgb8Gray, fGuided, 0.005, 8 );
                    //mats["crossFiltered8"].convertTo( mats["crossFiltered16"], CV_16UC1, 10001.f / 255.f );
                    //cv::imshow( CROSS_WINDOW_NAME, mats["crossFiltered8"] );
                    //cv::imshow( "guided16", mats["crossFiltered16"] );

                    cv::Mat fGuided255;
                    fGuided.convertTo( fGuided255, CV_8UC1 );
                    cv::imshow( "fGuided255", fGuided255 );

                    cv::Mat x1;
                    cv::addWeighted( rgb8Gray, .5, fGuided255, .5, .0, x1, CV_8UC1 );
                    cv::imshow( "overlay(rgb,guided)", x1 );
                }

                // IR8 + RGB8
                cv::Mat irAndRgb;
                if ( showIrAndRgb && !ir8.empty() && !rgb8.empty() )
                {
                    combineIRandRgb( ir8, rgb8, rgb8.size(), irAndRgb, .9 );
                    imshow( "irAndRgb", irAndRgb * 2.0 );
                }

                // DEP8 + RGB8
                cv::Mat dep8AndRgb;
                if ( showDep16AndRgb && !mapped8.empty() && !rgb8.empty() )
                {
                    std::vector<cv::Mat> rgb8s;
                    cv::split( rgb8, rgb8s );
                    cv::merge( (std::vector<cv::Mat>){rgb8s[0]*0.8f,rgb8s[1]*0.8f, mapped8 * 2.f}, dep8AndRgb );
                    imshow( "dep8AndRgb", dep8AndRgb );
                    std::cout << "dep8AndRgb showed..." << std::endl;
                }

                // IR8 + DEP8
                if ( showOffset && g_ir.IsGenerating() && !ir8.empty() && !dep8.empty() )
                {
                    std::cout << "starting irAndDep8" << std::endl;
                    cv::merge( (std::vector<cv::Mat>{dep8/2,dep8/2,ir8*10}), mats["irAndDep8"] );
                    std::cout << "finished irAndDep8" << std::endl;
                    imshow( "irAndDep8", mats["irAndDep8"] );

                    cv::Mat offsIr, offsDep8;
                    ir8.colRange(4,ir8.cols-4).copyTo( offsIr );
                    dep8.colRange(0,dep8.cols-8).copyTo( offsDep8 );
                    cv::merge( (std::vector<cv::Mat>{offsDep8/2,offsDep8/2,offsIr*10}), mats["offsIrAndDep8"] );
                    std::cout << "showed offsIrAndDep8" << std::endl;
                    imshow( "offsIrAndDep8", mats["offsIrAndDep8"] );

                    cv::Mat offs2Ir, offs2Dep8;
                    ir8.colRange(8,ir8.cols).copyTo( offs2Ir );
                    dep8.colRange(0,dep8.cols-8).copyTo( offs2Dep8 );
                    cv::merge( (std::vector<cv::Mat>{offs2Dep8/2,offs2Dep8/2,offs2Ir*10}), mats["offs2IrAndDep8"] );
                    std::cout << "showed offs2IrAndDep8" << std::endl;
                    imshow( "offs2IrAndDep8", mats["offs2IrAndDep8"] );
                }

                // Key Input
                if ( irGenerator.IsGenerating() )
                    c = cv::waitKey( 300 );
                else
                    c = cv::waitKey( 10 );

                switch ( c )
                {
                    case 32:
                        {
                            for ( auto it = mats.begin(); it != mats.end(); ++it )
                            {
                                am::CvImageDumper::Instance().dump( it->second, it->first, false );
                            }

                            am::CvImageDumper::Instance().dump( dep8,        "dep8",        false, "pgm" );
                            am::CvImageDumper::Instance().dump( dep16,       "dep16",       false, "pgm" );
                            am::CvImageDumper::Instance().dump( dep16,       "dep16",       false, "png" );
                            am::CvImageDumper::Instance().dump( mapped16,    "mapped16",    false, "png" );
                            am::CvImageDumper::Instance().dump( rgb8,        "img8",        false );
                            am::CvImageDumper::Instance().dump( rgb8_1280,        "img8_1280", false );
                            am::CvImageDumper::Instance().dump( ir8,         "ir8",         false );
                            am::CvImageDumper::Instance().dump( dep8AndRgb, "dep8AndRgb", false );
                            am::CvImageDumper::Instance().dump( irAndRgb,    "irAndRgb",    false  );
                            am::CvImageDumper::Instance().step();
                        }
                        break;

                    case 'd':
                        showDep8 = !showDep8;
                        std::cout << "showDep8: " << util::printBool(showDep8) << std::endl;
                        break;
                    case 'D':
                        showDep16AndRgb = !showDep16AndRgb;
                        std::cout << "showDep16AndRgb: " << util::printBool(showDep16AndRgb) << std::endl;
                        break;
                    case 'i':
                        toggleIR();
                        std::cout << "showIR: " << util::printBool(showIR) << std::endl;
                        break;
                    case 'I':
                        showIrAndRgb = !showIrAndRgb;
                        std::cout << "showIrAndRgb: " << util::printBool(showIrAndRgb) << std::endl;
                        break;
                    case 'o':
                        showOffset = !showOffset;
                        break;
                    case 'r':
                        showRgb = !showRgb;
                        std::cout << "showRgb: " << util::printBool(showRgb) << std::endl;
                        break;
                    case 'a':
                        toggleAltViewpoint();
                        break;
                    case 'g':
                        showGuided = !showGuided;
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
                    default:
                        std::cout << (int)c << std::endl;
                        break;
                }
            }
        }
} myPlayer;

//---------------------------------------------------------------------------
// CB
//---------------------------------------------------------------------------

void on_contrast_alpha_trackbar( int, void* )
{
    myPlayer.alpha = (double) myPlayer.alpha_slider / myPlayer.alpha_slider_max ;
}

void on_contrast_beta_trackbar( int, void* )
{
    myPlayer.beta = (double) myPlayer.beta_slider;// / myPlayer.beta_slider_max ;
}

void on_cross_gaussian_delta_trackbar( int, void* )
{
    myPlayer.cross_gaussian_delta.value = (double) myPlayer.cross_gaussian_delta.slider / myPlayer.cross_gaussian_delta.slider_max * 5.;
}

void on_cross_eucledian_delta_trackbar( int, void* )
{
    myPlayer.cross_eucledian_delta.value = (double) myPlayer.cross_eucledian_delta.slider / myPlayer.cross_eucledian_delta.slider_max ;
}

void on_cross_filter_range_trackbar( int, void* )
{
    myPlayer.cross_filter_range.value = myPlayer.cross_filter_range.slider;
}

void on_cross_filter_iterations_trackbar( int, void* )
{
    myPlayer.cross_filter_iterations.value = myPlayer.cross_filter_iterations.slider;
}

#define Yang 1
int testFiltering()
{
    //std::string path = "/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130725_1809/dep16_00000001.png_mapped.png";
    std::string path        = "/media/Storage/workspace_ubuntu/cpp_projects/KinfuSuperRes/BilateralFilteringCuda/build/bilFiltered_02.ppm";
    std::string guide_path  = "/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130725_1809/img8_00000001.png";

    /*TMatDict dict;
    dict["overlayMappedLarge"] = cv::imread( path + "overlayMappedLarge_00000003.png", -1 );
    dict["img8"] = cv::imread( path + "img8_00000003.png", -1 );
    dict["dep8"] = cv::imread( path + "dep8_00000003.png", -1 );
    dict["dep16"] = cv::imread( path + "dep16_00000003.png", -1 );
    dict["dep16MappedLarge"] = cv::imread( path + "dep16MappedLarge_00000003.png", -1 );
    imshow( "dep8", dict["dep8"] );
    imshow( "dep16", dict["dep16"] );*/
    cv::Mat dep16 = cv::imread( path, cv::IMREAD_UNCHANGED ), fDep16;
    if ( dep16.type() == CV_16UC1 )
    {
        std::cout << "dep16.type(): CV_16UC1" << std::endl;
        dep16.convertTo( fDep16, CV_32FC1, 255.0 / 10001.0 );
    }
    else if ( dep16.type() == CV_8UC1 )
    {
        std::cout << "dep16.type(): CV_8UC1" << std::endl;
        dep16.convertTo( fDep16, CV_32FC1 );
    }
    else if ( dep16.type() == CV_8UC3 )
    {
        fDep16.create( dep16.rows, dep16.cols, CV_32FC1 );
        std::cout << "dep16.type(): CV_8UC3" << std::endl;
        for ( int y = 0; y < dep16.rows; ++y )
            for ( int x = 0; x < dep16.cols; ++x )
        {
                fDep16.at<float>( y,x ) = ( (float)(dep16.at<cv::Vec3b>(y,x)[0]) );
        }
    }
    else
    {
        std::cout << "dep16 unknown type" << std::endl;
        return 1;
    }
    imshow( "fDep16", fDep16 );

    cv::Mat guide8 = cv::imread( guide_path, cv::IMREAD_UNCHANGED ), grayGuide8, fGuide8;
    cv::cvtColor( guide8, grayGuide8, cv::COLOR_RGB2GRAY );
    grayGuide8.convertTo( fGuide8, CV_32FC1 );

#if Yang
    qx_constant_time_bilateral_filter_published qx_bf;
    double sigma_spatial = 0.03;
    double sigma_range   = 0.05;

    qx_bf.init( dep16.rows, dep16.cols );

    double **in  = qx_allocd( dep16.rows, dep16.cols );
    util::CopyCvImgToDouble( fDep16, in );
    double **guide = qx_allocd( fGuide8.rows, fGuide8.cols );
    util::CopyCvImgToDouble( fGuide8, guide );

    double **out = qx_allocd( dep16.rows, dep16.cols );

    qx_bf.filter( out, in, sigma_spatial, sigma_range, guide );

    cv::Mat cvFiltered, cvFiltered255;
    util::CopyDoubleToCvImage( out, dep16.rows, dep16.cols, cvFiltered );
    cvFiltered.convertTo( cvFiltered255, CV_8UC1 );
    cv::imshow( "cvFiltered255", cvFiltered255 );
    cv::imwrite( path + "_qx_cross_filtered.png", cvFiltered255 );

    qx_freed( in  );
    qx_freed( out );
    qx_freed( guide );

#endif

    // sigc = 50;
    // double sc = sigc/10.0;
    // guidedFilterTBB(filledDepthf,srcImagef,filteredDepthf,d,(float)(sc*0.001),8);
    cv::Mat fGuided, fGuided255;
    Filtering::guidedFilterSrc1Guidance1( fDep16, fGuide8, fGuided, 0.005, 8 );
    fGuided.convertTo( fGuided255, CV_8UC1 );
    cv::imshow( "fGuided255", fGuided255 );

    cv::Mat x1;
    cv::addWeighted( fGuide8, .5, fGuided255, .5, .0, x1, CV_8UC1 );
    cv::imshow( "x1", x1 );

    cv::waitKey();
    return 0;

    cv::Mat dep8;
    cv::convertScaleAbs( dep16, dep8, 255.0 / 10001.0 );
    imshow( "dep8", dep8 );

    cv::Mat res_dep16, res_dep8;
    cv::resize( dep16, res_dep16, dep16.size(), 0, 0, cv::INTER_NEAREST );
    cv::convertScaleAbs( res_dep16, res_dep8, 255.0 / 10001.0 );
    cv::imshow( "res_dep8", res_dep8 );
    cv::waitKey(20);
    //return 0;

    double sigmaD = 20.0;
    double sigmaR = 50.0;

    static am::BilateralFiltering bf( sigmaD, sigmaR );
    cv::Mat cross16;
    //bf.runFilter<ushort,uchar>( dict["dep16MappedLarge"], dict["img8"], cross16 );
    bf.runFilter<ushort>( dep16, cross16 );
    imshow( "cross16", cross16 );

    char fname[256];
    sprintf( fname, "cross16_%f_%f.png", sigmaD, sigmaR );
    cv::imwrite( fname, cross16 );
    cv::Mat cross8;
    cv::convertScaleAbs( cross16, cross8, 255.0 / 10001.0 );
    sprintf( fname, "cross8_%f_%f.png", sigmaD, sigmaR );
    cv::imwrite( fname, cross8 );
    imshow( "cross8", cross8 );

    cv::waitKey(0);
}

void getCost( cv::Mat const& )
{

}

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/copy.h>
int testCostVolume()
{
    const int   L       = 20; // depth search range
    const float ETA     = .5f;
    const float ETA_L_2 = SQR(ETA*L);
    const float MAXRES  = 255.0f;

    // read
    //std::string path    = "/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130809_1415/";
    std::string path    = "/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130809_1438/";
    cv::Mat dep16       = cv::imread( path + "mapped16_00000001.png", cv::IMREAD_UNCHANGED );
    cv::Mat rgb8        = cv::imread( path + "img8_1280_00000001.png", cv::IMREAD_UNCHANGED );
    cv::Mat dep8;         dep16.convertTo( dep8, CV_8UC1, 255.f / 10001.f );
    cv::Mat dep8_large;

    // show
    cv::imshow( "dep16", dep16 );
    cv::imshow( "dep8", dep8 );
    cv::imshow( "img8" , rgb8  );

    char key_pressed = 0;

    // crossFiltering
    static BilateralFilterCuda<float> cbfc;
    cbfc.setFillMode( FILL_ONLY_ZEROS );

    // select input depth
    float depMax = 255.f;
    cv::Mat &dep = dep8; // 640 by default
    // upscale
    if ( rgb8.size() != dep16.size() )
    {
        cv::resize( dep16, dep8_large, rgb8.size(), 0, 0, cv::INTER_NEAREST );
        dep = dep8_large;
        depMax = 10001.f;
    }

    // convert input depth to float
    cv::Mat fDep; dep.convertTo( fDep, CV_32FC1  );

    BilateralFilterCuda<float> bfc;
    bfc.setIterations( 5 );
    bfc.setFillMode( FILL_ONLY_ZEROS );
    cv::Mat bilfiltered;
    bfc.runBilateralFiltering( fDep, rgb8, bilfiltered,
                               5.f, .1f, 10, 1.f );
    cv::imshow( "bilf", bilfiltered / depMax );
    bilfiltered.copyTo( fDep );

#if 1
    YangFiltering::run( fDep, rgb8, fDep );
#elif 0
    // input: fDep(CV_32FC1,0..10001.f), rgb8(CV_8UC3)

    // prepare
    cv::Mat truncC2     ( fDep.size(), CV_32FC1 ); // C(d)
    cv::Mat truncC2_prev( fDep.size(), CV_32FC1 ); // C(d-1)
    cv::Mat minDs       ( fDep.size(), CV_32FC1 ); // d_min
    cv::Mat minC        ( fDep.size(), CV_32FC1 ); // C(d_min)
    cv::Mat minCm1      ( fDep.size(), CV_32FC1 ); // C(d_min-1)
    cv::Mat minCp1      ( fDep.size(), CV_32FC1 ); // C(d_min+1)

    for ( int it = 0; it < 5; ++it )
    {
        // select range of candidates
        double maxVal;
        cv::minMaxIdx( fDep, 0, &maxVal );
        std::cout << "max: " << maxVal << std::endl;

        minC  .setTo( maxVal * maxVal );
        minCm1.setTo( maxVal * maxVal );
        minCp1.setTo( maxVal * maxVal );

        for ( int d = 0; d < min((float)maxVal + L + 1, MAXRES); d+=1 )
        {
            std::cout << "d: " << d << " -> " << maxVal + L + 1 << std::endl;

            // calculate truncated cost
            MyThrustUtil::squareDiff( fDep, d, truncC2, ETA_L_2 );

            // filter cost slice
            cbfc.runBilateralFiltering( /*            in: */ truncC2,
                                        /*         guide: */ rgb8,
                                        /*           out: */ truncC2,
                                        /* spatial sigma: */ 1.5f,
                                        /*   range sigma: */ .03f,
                                        /*  kernel range: */ 12 );

            // track minimums
            MyThrustUtil::minMaskedCopy( truncC2_prev, truncC2, d, minC, minDs, minCm1, minCp1 );

            // show
            //cv::imshow( "minC" , minC / MAXRES / MAXRES );
            //cv::imshow( "minDs", minDs / depMax );

            truncC2.copyTo( truncC2_prev );

            //cv::waitKey(50);
        }

        // refine minDs based on C(d_min), C(d_min-1), C(d_min+1)
        MyThrustUtil::subpixelRefine( minC, minCm1, minCp1, minDs );

        // copy to output
        minDs.copyTo( fDep );
        cv::imshow( "fDep", fDep / MAXRES );
        cv::waitKey(50);
    }

    // output: fDep
#else
    // prepare output
    cv::Mat fDep_next( dep.rows, dep.cols, CV_32FC1 );
    // iterate
    for ( int it = 0; it < 5; ++it )
    {
        // select range of candidates
        double maxVal;
        cv::minMaxIdx( fDep, 0, &maxVal );
        std::cout << "max: " << maxVal << std::endl;

        // calculate cost volume for every depth candidate
        cv::Mat C      ( dep.size(), CV_32FC1 );                                // simple depth difference
        cv::Mat C2     ( dep.size(), CV_32FC1 );                                // squared depth difference
        cv::Mat truncC2( dep.size(), CV_32FC1 );                                // truncated squared depth difference
        cv::Mat minC   ( dep.size(), CV_32FC1 ); minC.setTo( maxVal * maxVal ); // minimum cost over d values
        cv::Mat minDs  ( dep.size(), CV_32FC1 );                                // d values for minimum costs
        for ( int d = 0; d < maxVal + L + 1; d+=1 )
        {
            // info
            std::cout << d << std::endl;

            // calculate cost slice
            cv::absdiff( fDep, d, C );
            cv::multiply( C, C, C2 );
            truncC2 = cv::min( C2, ETA_L_2 );

            // filter cost slice
            cbfc.runBilateralFiltering( /*            in: */ truncC2,
                                        /*         guide: */ rgb8,
                                        /*           out: */ truncC2,
                                        /* spatial sigma: */ 1.5f,
                                        /*   range sigma: */ .03f,
                                        /*  kernel range: */ 12 );
            //cv::imshow( "C2", C2 / 65536.f );
            //cv::imshow( "truncC2", truncC2 / ETA_L_2 / 2.f );

            /// track minimum cost d values
            {
                // replace minimum costs
                minC = cv::min( /*  stored: */ minC,
                                /* current: */ truncC2 );

                // selection of minimum places
                // minMask = (minC == truncC2)
                cv::Mat minMask;
                cv::compare( /*      new costs: */ minC,
                             /*  current costs: */ truncC2,
                             /* current places: */ minMask,
                             /* condition "==": */ CV_CMP_EQ );

                // minDepths( minC == truncC2 ) = current d;
                minDs.setTo( d, minMask );
            }

            key_pressed = cv::waitKey(7);
            if ( key_pressed == 27 )
                break;
        }
        cv::imshow( "minC" , minC / MAXRES / MAXRES );
        cv::imshow( "minDs", minDs );

        /// calculate costs of neighbour depths
        cv::Mat ftmp( dep.size(), CV_32FC1 );
        // d_-
        cv::Mat d_m1( dep.size(), CV_32FC1 );
        cv::subtract( minDs, 1.f, d_m1, cv::Mat(), CV_32FC1 );
        // d_+
        cv::Mat d_p1( dep.size(), CV_32FC1 );
        cv::add( minDs, 1.f, d_p1, cv::Mat(), CV_32FC1 );

        // f(d_-)
        cv::Mat f_d_m1;
        cv::absdiff( fDep, d_m1, ftmp );
        cv::multiply( ftmp, ftmp, f_d_m1 );
        f_d_m1 = cv::min( f_d_m1, ETA_L_2 );

        // f(d_+)
        cv::Mat f_d_p1( dep.rows, dep.cols, CV_32FC1 );
        cv::absdiff( fDep, d_p1, ftmp );
        cv::multiply( ftmp, ftmp, f_d_p1 );
        f_d_p1 = cv::min( f_d_p1, ETA_L_2 );

        /// subpixel
        cv::Mat a1 = ( f_d_p1 - f_d_m1 );
        cv::Mat a2 = ( 2.f * (f_d_p1 + f_d_m1 - 2.f * minC) );
        cv::Mat a3;
        cv::divide( a1, a2, a3, 1.0, CV_32FC1 );
        a3 = cv::min( a3, MAXRES/16.f );
        a3 = cv::max( -(MAXRES/16.f), a3 );
        cv::subtract( minDs, a3, fDep_next, cv::Mat(), CV_32FC1 );
#if 0
        //cv::imwrite( "a1", a1 );
        std::string dpath = "out/";
        util::writeCvMat2MFile<float>( a1       , dpath+"load_a1.m"       , "a1" );
        util::writeCvMat2MFile<float>( a2       , dpath+"load_a2.m"       , "a2" );
        util::writeCvMat2MFile<float>( a3       , dpath+"load_a3.m"       , "a3" );
        util::writeCvMat2MFile<float>( minDs    , dpath+"load_minDs.m"    , "minDs" );
        util::writeCvMat2MFile<float>( fDep_next, dpath+"load_fDep_next.m", "fDep_next" );
#endif


        {
            double minVal, maxVal;
            cv::minMaxIdx( fDep_next, &minVal, &maxVal );
            std::cout << "minVal(fDep_next): " << minVal << ", "
                      << "maxVal(fDep_next): " << maxVal << std::endl;
        }

        cv::imshow( "fDep_next", fDep_next / MAXRES );
        cv::waitKey(10);
        fDep_next.copyTo( fDep );
    }
#endif

    {
        double minVal, maxVal;
        cv::minMaxIdx( fDep, &minVal, &maxVal );
        std::cout << "minVal(fDep): " << minVal << ", "
                  << "maxVal(fDep): " << maxVal << std::endl;
        cv::Mat tmp;
        fDep.convertTo( tmp, CV_16UC1, 10001.f / MAXRES );
        cv::imwrite( "yang16.png", tmp, (std::vector<int>){cv::IMWRITE_PNG_COMPRESSION,0} );
    }

    while ( key_pressed != 27 )
    {
        key_pressed = cv::waitKey();
    }

    return EXIT_SUCCESS;
}

int main( int argc, char* argv[] )
{
    return testCostVolume();
    //MyCVPlayer::run();
    //return 0;

    //testFiltering();
    //return 0;

    //mainYet( argc, argv );
    //return 0;

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
                //util::catFile( std::string(path) + "/" + SAMPLE_XML_PATH );
                //rc = g_context.InitFromXmlFile( SAMPLE_XML_PATH, &errors );
                break;
            }
    }

#define RGB_WIDTH 1280
#define RGB_HEIGHT 1024
#define RGB_FPS 15
#if 1
    /// init NODES
    XnMapOutputMode modeIR;
    modeIR.nFPS = 30;
    modeIR.nXRes = 640;
    modeIR.nYRes = 480;
    XnMapOutputMode modeVGA;
    modeVGA.nFPS = RGB_FPS;
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
    rc = g_image.StartGenerating();
    CHECK_RC(rc, "Start generating RGB");

    //IR node creation
    rc = g_ir.Create(g_context);
    CHECK_RC(rc, "Create ir generator");
    rc = g_ir.SetMapOutputMode(modeIR);
    CHECK_RC(rc, "IR SetMapOutputMode XRes for 640, YRes for 480 and FPS for 30");
    //rc = g_ir.StartGenerating();
    //CHECK_RC(rc, "Start generating IR");
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

#endif

    /// INFO
    {
        xn::DepthMetaData depthMD;
        g_depth.GetMetaData( depthMD );
        std::cout << "depthMD.ZRes(): " << depthMD.ZRes() << std::endl;
    }
    //return 0;

    /// RUN
    //MyPlayer myPlayer;
    //myPlayer.toggleAltViewpoint();

    myPlayer.playGenerators( g_context, g_depth, g_image, g_ir );

    return EXIT_SUCCESS;
}
