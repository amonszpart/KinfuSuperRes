#include "CvImageDumper.h"

#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include "XnCppWrapper.h"
#include "../amCommon.h"
#include <iostream>

namespace am
{

    CvImageDumper::CvImageDumper()
        : frameID( 0 ), outputPath( "" )
    {
    }

    int CvImageDumper::dump( const cv::Mat &img, std::string title, bool step, std::string extension )
    {
        if ( img.empty() )
        {
            std::cerr << title << " empty, not dumping..." << std::endl;
            return 1;
        }

        XnBool doesExist = true;
        XnStatus rc = xnOSDoesDirecotyExist( outputPath.c_str(), &doesExist );
        CHECK_RC( rc, "xnOsDirectoryExist failed" );
        if ( !doesExist )
            xnOSCreateDirectory( outputPath.c_str() );

        std::cout << "img: " << img.cols << "x" << img.rows << std::endl;
        char name[256];
        sprintf( name, ("%s/%s_%08d." + extension).c_str(), outputPath.c_str(), title.c_str(), frameID );
        std::vector<int> imwrite_params;

        if ( extension == "png" )
        {
            imwrite_params.push_back( cv::IMWRITE_PNG_COMPRESSION );
            imwrite_params.push_back( 0 );
        }
        else if ( (extension == "jpg") || (extension == "jpeg") )
        {
            imwrite_params.push_back( cv::IMWRITE_JPEG_QUALITY );
            imwrite_params.push_back( 100 );
        }
        cv::imwrite( name, img, imwrite_params);

        std::cout << "dumped " << name << "..." << std::endl;

        //sprintf(title, "col8_%08d.png", framecount++);
        //cv::imwrite( title, cvImg );

        if ( step )
            ++frameID;

        return EXIT_SUCCESS;
    }

    int CvImageDumper::setOutputPath( std::string path )
    {
        time_t rawtime;
        struct tm * timeinfo;
        char buffer [80];

        time (&rawtime);
        timeinfo = localtime (&rawtime);

        strftime ( buffer, 80, "_%Y%m%d_%H%M", timeinfo );

        outputPath = path + std::string( buffer );

        return EXIT_SUCCESS;
    }

} // ns am
