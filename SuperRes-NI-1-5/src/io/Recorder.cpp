#include "Recorder.h"

//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#include <XnCppWrapper.h>

#include "../util/MaUtil.h"
#include "opencv2/opencv.hpp"
#include "CvImageDumper.h"
#include "../util/XnVUtil.h"

//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------

namespace am
{

    Recorder::Recorder( const std::string& recPath, std::string const samplePath )
        : _recPath( recPath ), _sample_path( samplePath ), _altViewpoint( false )
    {
    }

    Recorder::~Recorder()
    {
        context.Release();
        imageGenerator.Release();
        depthGenerator.Release();
        irGenerator.Release();
    }

    int Recorder::autoConfig()
    {
        using namespace xn;
        XnStatus nRetVal = XN_STATUS_OK;

        EnumerationErrors errors;

        printf("Reading config from: '%s'\n", _sample_path.c_str() );
        nRetVal = context.InitFromXmlFile( _sample_path.c_str(), &errors );
        CHECK_RC( nRetVal, "InitFromXmlFile" );

        if (nRetVal == XN_STATUS_NO_NODE_PRESENT)
        {
            XnChar strError[1024];
            errors.ToString(strError, 1024);
            printf("%s\n", strError);
            return (nRetVal);
        }
        else if (nRetVal != XN_STATUS_OK)
        {
            printf("Open failed: %s\n", xnGetStatusString(nRetVal));
            return (nRetVal);
        }

        // get the list of all created nodes
        NodeInfoList nodes;
        nRetVal = context.EnumerateExistingNodes(nodes);
        CHECK_RC(nRetVal, "Enumerate nodes");
        for ( auto it = nodes.Begin(); it != nodes.End(); it++ )
        {
            std::cout << "Node: " << (*it).GetInstanceName() << std::endl;
        }



        // create recorder
        nRetVal = recorder.Create(context);
        CHECK_RC(nRetVal, "Create recorder");
        // ouptut path
        nRetVal = recorder.SetDestination(XN_RECORD_MEDIUM_FILE, _recPath.c_str() );
        CHECK_RC(nRetVal, "Set recorder destination file");

        //DepthGenerator depthGenerator;
        nRetVal = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator );
        CHECK_RC(nRetVal, "Find depth generator");
        nRetVal = recorder.AddNodeToRecording( depthGenerator );
        CHECK_RC(nRetVal, "Add depth node to recording");

        //ImageGenerator imageGenerator;
        nRetVal = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator );
        CHECK_RC(nRetVal, "Find image generator");
        nRetVal = recorder.AddNodeToRecording( imageGenerator );
        CHECK_RC(nRetVal, "Add image node to recording");

        /*IRGenerator irGenerator;
        nRetVal = context.FindExistingNode(XN_NODE_TYPE_IR, irGenerator );
        CHECK_RC(nRetVal, "Find IR generator");
        nRetVal = recorder.AddNodeToRecording( irGenerator );
        CHECK_RC(nRetVal, "Add IR node to recording");*/

        return nRetVal;
    }

    int Recorder::run( bool displayColor )
    {
        using namespace xn;
        XnStatus nRetVal = XN_STATUS_OK;

        DepthMetaData depthMD;

        // Alternative viewpoint
        XnBool isSupported = depthGenerator.IsCapabilitySupported( "AlternativeViewPoint" );
        if ( isSupported )
        {
            XnStatus res = XN_STATUS_OK;
            if ( _altViewpoint )
            {
                res = imageGenerator.GetAlternativeViewPointCap().SetViewPoint( depthGenerator );
                if ( XN_STATUS_OK == res )
                {
                     printf("ImageGenerator.alternativeViewPoint set to depthgenerator's: %s\n", xnGetStatusString(res));
                }
                else
                {
                     printf("ImageGenerator.alternativeViewPoint NOT set to depthgenerator's: %s\n", xnGetStatusString(res));
                }
            }
            else
            {
                res = depthGenerator.GetAlternativeViewPointCap().ResetViewPoint();
                if ( XN_STATUS_OK == res )
                {
                     printf("NO alternative viewpoint set successfully: %s\n", xnGetStatusString(res));
                }
                else
                {
                     printf("Tried to set alternative viewpoint, but failed...: %s\n", xnGetStatusString(res));
                }
            }

            if ( XN_STATUS_OK != res )
            {
                printf("Something with AlternativeViewPoint failed: %s\n", xnGetStatusString(res));
            }
        }
        else
        {
            printf("Alternative viewpoint NOT supported...");
        }

        // declare CV
        cv::Mat cvDepth8, cvImg, cvDepth16;

        char c = 0;
        while ( (!xnOSWasKeyboardHit()) && (c != 27) )
        {
            // read DEPTH
            nRetVal = context.WaitAnyUpdateAll();
            if (nRetVal != XN_STATUS_OK)
            {
                printf("UpdateData failed: %s\n", xnGetStatusString(nRetVal));
                continue;
            }

            // convert depth
            util::nextDepthToMats( depthGenerator, &cvDepth8, &cvDepth16 );
            cv::imshow( "cvDepth8", cvDepth8 );
            util::nextImageAsMat( imageGenerator, &cvImg );

            if ( displayColor )
            {
                cv::imshow("cvImg", cvImg);
            }

            //CvImageDumper::Instance().dump( cvImg, "img8", false );
            //CvImageDumper::Instance().dump( cvDepth8, "dep8", false );
            //CvImageDumper::Instance().dump( cvDepth16, "dep16", true );

            c = cv::waitKey(5);
        }

        return 0;
    }

    void Recorder::setAltViewpoint( bool altViewPoint )
    {
        this->_altViewpoint = altViewPoint;
    }

    void Recorder::setSamplePath( std::string const& samplePath )
    {
        this->_sample_path = samplePath;
    }

    int Recorder::manualConfig( int vgaWidth, int irWidth, int vgaHeight, int irHeight )
    {
        XnStatus rc = XN_STATUS_OK;

        /// init NODES
        XnMapOutputMode modeVGA;
        modeVGA.nXRes = vgaWidth;
        modeVGA.nYRes = (vgaHeight < 0) ?
                            ((vgaWidth == 1280) ? 1024 : 480) :
                                vgaHeight;
        modeVGA.nFPS = (vgaWidth == 640) ? 30 : 15;
        XnMapOutputMode modeIR;
        modeIR.nXRes = irWidth;
        modeIR.nYRes = (irHeight < 0) ?
                           ((irWidth == 1280) ? 1024 : 480) :
                               irHeight;;
        modeIR.nFPS = (irWidth == 640) ? 30 : 15;
        printf( "Initing Recorder{ VGA(%d,%d,%d), IR(%d,%d,%d) }...\n",
                modeVGA.nXRes, modeVGA.nYRes, modeVGA.nFPS,
                modeIR.nXRes, modeIR.nYRes, modeIR.nFPS );

        //context inizialization
        rc = context.Init();
        CHECK_RC(rc, "Initialize context");

        // create recorder
        rc = recorder.Create(context);
        CHECK_RC(rc, "Create recorder");
        rc = recorder.SetDestination(XN_RECORD_MEDIUM_FILE, _recPath.c_str() );
        CHECK_RC(rc, "Set recorder destination file");

        //depth node creation
        rc = depthGenerator.Create(context);
        CHECK_RC(rc, "Create depth generator");
        rc = recorder.AddNodeToRecording( depthGenerator );
        CHECK_RC(rc, "DepthGenerator add to recording");
        rc = depthGenerator.StartGenerating();
        CHECK_RC(rc, "Start generating Depth");

        //RGB node creation
        rc = imageGenerator.Create(context);
        CHECK_RC(rc, "Create rgb generator");
        rc = imageGenerator.SetMapOutputMode(modeVGA);
        CHECK_RC(rc, "Depth SetMapOutputMode XRes for 1280, YRes for 1024 and FPS for 15");
        rc = imageGenerator.StartGenerating();
        CHECK_RC(rc, "Start generating RGB");
        rc = recorder.AddNodeToRecording( imageGenerator );
        CHECK_RC(rc, "imageGenerator add to recording");

        //IR node creation
        rc = irGenerator.Create(context);
        CHECK_RC(rc, "Create ir generator");
        rc = irGenerator.SetMapOutputMode(modeIR);
        CHECK_RC(rc, "IR SetMapOutputMode XRes for 640, YRes for 480 and FPS for 30");
        //rc = ir.StartGenerating();
        //CHECK_RC(rc, "Start generating IR");
    }


} // ns am
