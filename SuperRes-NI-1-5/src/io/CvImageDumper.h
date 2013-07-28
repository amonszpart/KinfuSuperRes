#ifndef CVIMAGEDUMPER_H
#define CVIMAGEDUMPER_H

#include <opencv2/core/core.hpp>

namespace am
{

    class CvImageDumper
    {
        public:

            static CvImageDumper& Instance()
            {
                static CvImageDumper instance;
                return instance;
            }

            int dump( cv::Mat const& img, std::string title, bool step = true );
            int setOutputPath( std::string path );
            int step() { ++frameID; };

        protected:
            unsigned long frameID;
            std::string outputPath;

            CvImageDumper();

    };

} // ns am

#endif // CVIMAGEDUMPER_H
