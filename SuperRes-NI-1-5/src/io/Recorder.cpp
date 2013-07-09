#include "Recorder.h"

//---------------------------------------------------------------------------
// Includes
//---------------------------------------------------------------------------
#include <XnCppWrapper.h>

#include "../util/MaUtil.h"
#include "opencv2/opencv.hpp"
#include "CvImageDumper.h"

//---------------------------------------------------------------------------
// Code
//---------------------------------------------------------------------------

namespace am
{

    Recorder::Recorder( const std::string& recPath, std::string const samplePath )
        : _recPath( recPath ), _sample_path( samplePath ), _altViewpoint( true )
    {
    }

    int Recorder::run( bool displayColor )
    {
        using namespace xn;
        XnStatus nRetVal = XN_STATUS_OK;

        Context context;
        EnumerationErrors errors;
        DepthMetaData depthMD;

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

        // create recorder
        xn::Recorder recorder;
        nRetVal = recorder.Create(context);
        CHECK_RC(nRetVal, "Create recorder");

        nRetVal = recorder.SetDestination(XN_RECORD_MEDIUM_FILE, _recPath.c_str() );
        CHECK_RC(nRetVal, "Set recorder destination file");

        DepthGenerator depthGenerator;
        nRetVal = context.FindExistingNode(XN_NODE_TYPE_DEPTH, depthGenerator );
        CHECK_RC(nRetVal, "Find depth generator");
        nRetVal = recorder.AddNodeToRecording( depthGenerator );
        CHECK_RC(nRetVal, "Add depth node to recording");

        ImageGenerator imageGenerator;
        nRetVal = context.FindExistingNode(XN_NODE_TYPE_IMAGE, imageGenerator );
        CHECK_RC(nRetVal, "Find image generator");
        nRetVal = recorder.AddNodeToRecording( imageGenerator );
        CHECK_RC(nRetVal, "Add image node to recording");

        /*IRGenerator irGenerator;
        nRetVal = context.FindExistingNode(XN_NODE_TYPE_IR, irGenerator );
        CHECK_RC(nRetVal, "Find IR generator");
        nRetVal = recorder.AddNodeToRecording( irGenerator );
        CHECK_RC(nRetVal, "Add IR node to recording");*/


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

            CvImageDumper::Instance().dump( cvImg, "img8", false );
            CvImageDumper::Instance().dump( cvDepth8, "dep8", true );
            CvImageDumper::Instance().dump( cvDepth16, "dep16", true );

            c = cv::waitKey(5);
        }

        depthGenerator.Release();
        imageGenerator.Release();
        recorder.Release();
        context.Release();

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

} // ns am
