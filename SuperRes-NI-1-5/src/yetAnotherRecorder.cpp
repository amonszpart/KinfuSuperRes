//---------------------------------------------------------------------------

// Includes

//---------------------------------------------------------------------------

#include "util/XnVUtil.h"

#include <iostream>
#include <XnOpenNI.h>
#include <XnLog.h>
#include <XnCppWrapper.h>

#include <math.h>
#include <XnOS.h>

#include <cxcore.h>
#include <highgui.h>
#include <cv.h>
#include <ml.h>

using namespace xn;
using namespace std;

//---------------------------------------------------------------------------

// Defines

//---------------------------------------------------------------------------

#define GL_WIN_SIZE_X 1280
#define GL_WIN_SIZE_Y 1024

#define RGB_WIDTH 1280
#define RGB_HEIGHT 1024

#define DISPLAY_MODE_OVERLAY        1
#define DISPLAY_MODE_DEPTH                2
#define DISPLAY_MODE_IMAGE                3
#define DEFAULT_DISPLAY_MODE        DISPLAY_MODE_DEPTH

#define MAX_DEPTH 10000

enum Visualization{
    RGB,
    IR
};

//---------------------------------------------------------------------------
// Globals
//---------------------------------------------------------------------------
//nodes
Context context;
DepthGenerator depthGen;
ImageGenerator rgbGen;
IRGenerator irGen;

//images
IplImage *depthImage = cvCreateImage( cvSize(640,480), 16, 1);
IplImage *irImage    = cvCreateImage( cvSize(640,480), 16, 1);
IplImage *rgbImage   = cvCreateImage( cvSize(RGB_WIDTH,RGB_HEIGHT), 8, 3);

//raw data buffers
const XnDepthPixel* depthMap;
const XnIRPixel* irMap;
const XnUInt8* rgbMap;

//auxiliary buffers and images
XnDepthPixel tempDepth[640*480];
XnDepthPixel tempIR[640*480];
IplImage *bgrImage = cvCreateImage( cvSize(RGB_WIDTH,RGB_HEIGHT), 8, 3);
XnDepthPixel** help = (XnDepthPixel**)&depthMap;

CvPoint selection;
Visualization vis = RGB;
const char viewsPrefix[] = "views/";
unsigned int capturedViews = 0;

//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------
void mouseCB ( int event, int x, int y, int flags, void* param )
{
    if ( event == CV_EVENT_LBUTTONDOWN )
    {
        selection.x = x;
        selection.y = y;
        cout << "x = " << selection.x << "; y =  " << selection.y << " depth = " << (*((XnDepthPixel**)param))[640*y + x] << "\n";
    }
}

void switchRGBIR()
{
    switch ( vis )
    {
        case RGB:
            {
                vis = IR;
                rgbGen.StopGenerating();
                irGen.StartGenerating();
                break;
            }
        case IR:
            {
                vis = RGB;
                irGen.StopGenerating();
                rgbGen.StartGenerating();
                break;
            }
    }
}

void capture()
{
    if ( vis == IR )
    {
        switchRGBIR(/*rgbGen, irGen*/);
    }

    char filename[100];
    sprintf( filename, "%sview%04d/color.png", viewsPrefix, capturedViews );
    printf("%s\n", filename);

    context.WaitOneUpdateAll( rgbGen );

    rgbMap = rgbGen.GetImageMap();
    cvSetData(rgbImage, (void*)rgbMap, rgbImage->widthStep);
    cvCvtColor(rgbImage, bgrImage, CV_RGB2BGR);
    cvSaveImage(filename, bgrImage);

    switchRGBIR();

    sprintf(filename, "%sview%04d/ir.png", viewsPrefix, capturedViews);
    printf("%s\n", filename);

    context.WaitOneUpdateAll(irGen);
    irMap = irGen.GetIRMap();
    unsigned int max = 0;
    for(int i = 0; i < 640*480; ++i){
        if(irMap[i] > max){
            max = irMap[i];
        }
    }
    for(int i = 0; i < 640*480; ++i){
        tempIR[i] = (int)((double)irMap[i]/max*65535);
    }
    cvSetData(irImage, (void*)tempIR, irImage->widthStep);
    cvSaveImage(filename, irImage);

    capturedViews++;
}

int mainYet(int argc, char* argv[]){
    XnStatus nRetVal = XN_STATUS_OK;

    XnMapOutputMode modeIR;
    modeIR.nFPS = 30;
    modeIR.nXRes = 640;
    modeIR.nYRes = 480;
    XnMapOutputMode modeVGA;
    modeVGA.nFPS = 15;
    modeVGA.nXRes = RGB_WIDTH;
    modeVGA.nYRes = RGB_HEIGHT;

    //context inizialization
    nRetVal = context.Init();
    CHECK_RC(nRetVal, "Initialize context");

    //depth node creation
    nRetVal = depthGen.Create(context);
    CHECK_RC(nRetVal, "Create depth generator");
    nRetVal = depthGen.StartGenerating();
    CHECK_RC(nRetVal, "Start generating Depth");

    //RGB node creation
    nRetVal = rgbGen.Create(context);
    CHECK_RC(nRetVal, "Create rgb generator");
    nRetVal = rgbGen.SetMapOutputMode(modeVGA);
    CHECK_RC(nRetVal, "Depth SetMapOutputMode XRes for 240, YRes for 320 and FPS for 30");
    nRetVal = rgbGen.StartGenerating();
    CHECK_RC(nRetVal, "Start generating RGB");

    //IR node creation
    nRetVal = irGen.Create(context);
    CHECK_RC(nRetVal, "Create ir generator");
    nRetVal = irGen.SetMapOutputMode(modeIR);
    CHECK_RC(nRetVal, "Depth SetMapOutputMode XRes for 640, YRes for 480 and FPS for 30");

    cvNamedWindow("Distance", CV_WINDOW_AUTOSIZE);
    cvSetMouseCallback("Distance", mouseCB, help);

    while(true){
        nRetVal = context.WaitOneUpdateAll(depthGen);

        if (nRetVal == XN_STATUS_OK) {
            depthMap = depthGen.GetDepthMap();
            switch(vis){
                case RGB:{
                        rgbMap = rgbGen.GetImageMap();
                        break;
                    }
                case IR:{
                        irMap = irGen.GetIRMap();
                        break;
                    }
            }

            for(int i = 0; i < 640*480; ++i){
                tempDepth[i] = (int)((double)(MAX_DEPTH - depthMap[i])/
                                     MAX_DEPTH*65535);
            }
            cvSetData(depthImage, (void*)tempDepth, depthImage->widthStep);
            switch(vis){
                case RGB:{
                        cvSetData(rgbImage, (void*)rgbMap, rgbImage->widthStep);
                        cvCvtColor(rgbImage, bgrImage, CV_RGB2BGR);
                        break;
                    }
                case IR:{
                        unsigned int max = 0;
                        for(int i = 0; i < 640*480; ++i){
                            if(irMap[i] > max){
                                max = irMap[i];
                            }
                        }
                        for(int i = 0; i < 640*480; ++i){
                            tempIR[i] = (int)((double)irMap[i]/max*65535);
                        }
                        cvSetData(irImage, (void*)tempIR, irImage->widthStep);
                        break;
                    }
            }

            cvShowImage("Depth", depthImage);
            switch(vis){
                case RGB:{
                        cvShowImage( "RGB", bgrImage );
                        //doOverlay( depthImage, bgrImage );
                        break;
                    }
                case IR:{
                        cvShowImage( "IR", irImage );
                        break;
                    }
            }



            char c = cvWaitKey(33);
            switch(c){
                case 's':{
                        switchRGBIR();
                        break;
                    }
                case 'c':{
                        capture();
                        break;
                    }
                case 27:{
                        break;
                    }
            }
            if( c == 27 ) break;
        }

        else{
            printf("Failed updating data: %s\n", xnGetStatusString(nRetVal));
        }
    }

    cvDestroyWindow( "Distance");
    context.Release();

    return 0;

}
