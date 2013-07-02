#ifndef RECORDINGTEST_H
#define RECORDINGTEST_H

#include <string>

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

        public:
            Recorder( const std::string& recPath, const std::string samplePath );
            int run( bool displayColor = true );

            void setAltViewpoint( bool altViewPoint );
            void setSamplePath( std::string const& samplePath );
    };

} // ns am

#endif // RECORDINGTEST_H
