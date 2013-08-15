#ifndef RECORDINGTEST_H
#define RECORDINGTEST_H

#include <string>
#include <XnCppWrapper.h>

//---------------------------------------------------------------------------
// Defines
//---------------------------------------------------------------------------
//#define SAMPLE_XML_PATH "/home/bontius/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/Samples/Config/SamplesConfig.xml"

namespace am
{

    class Recorder
    {
            std::string _recPath;
            std::string _sample_path;
            bool _altViewpoint;

            xn::Context context;
            xn::DepthGenerator depthGenerator;
            xn::ImageGenerator imageGenerator;
            xn::IRGenerator irGenerator;
            xn::Recorder recorder;

        public:
            Recorder( const std::string& recPath, const std::string samplePath );
            ~Recorder();
            int run( bool displayColor = true );

            void setAltViewpoint( bool altViewPoint );
            void setSamplePath( std::string const& samplePath );
            int manualConfig(int vgaWidth = 640, int irWidth = 640 , int vgaHeight = -1, int irHeight = -1 );
            int autoConfig();
    };

} // ns am

#endif // RECORDINGTEST_H
