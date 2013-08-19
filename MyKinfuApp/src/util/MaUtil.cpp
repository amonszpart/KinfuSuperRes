#include "MaUtil.h"

/** Project includes */
#include "XnCppWrapper.h"
#include "opencv2/imgproc/imgproc.hpp"

/** opencv includes */
#include "opencv2/highgui/highgui.hpp"

/** STD includes */
#include <iostream>

namespace util
{

    unsigned int getClosestPowerOfTwo( unsigned int n )
    {
        unsigned int m = 2;
        while ( m < n )
            m <<= 1;

        return m;
    }

    int cvMatFromXnDepthMetaData( const xn::DepthMetaData &depthMD, cv::Mat* pCvDepth8, cv::Mat* pCvDepth16 )
    {
        static constexpr double conversionRatio = 255.0/2047.0;

        // CHECK
        if ( (pCvDepth8 == nullptr) && (pCvDepth16 == nullptr) )
            return -1;

        // INIT
        cv::Mat cvDepth16;
        cvDepth16.create( depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1 );

        // COPY
        {
            const XnDepthPixel *pDepthPixels = depthMD.Data();
            int offset = 0;
            for ( XnUInt y = 0; y < depthMD.YRes(); ++y, offset += depthMD.XRes() )
                memcpy( cvDepth16.data + cvDepth16.step * y, pDepthPixels + offset, depthMD.XRes() * sizeof(XnDepthPixel) );
        }

        // OUTPUT
        {
            if (pCvDepth8 != nullptr)
            {
                pCvDepth8->create( depthMD.FullYRes(), depthMD.FullXRes(), CV_8UC1 );
                cvDepth16.convertTo( *pCvDepth8, CV_8UC1, conversionRatio );
            }
            if (pCvDepth16 != nullptr)
            {
                pCvDepth16->create( depthMD.FullYRes(), depthMD.FullXRes(), CV_16UC1 );
                cvDepth16.copyTo( *pCvDepth16 );
            }
        }

        return 0;
    }

    // depth to CV::MAT
    int nextDepthToMats( xn::DepthGenerator& g_DepthGenerator, cv::Mat* pCvDepth8, cv::Mat* pCvDepth16 )
    {
        // CHECK
        if ( (pCvDepth8 == nullptr) && (pCvDepth16 == nullptr) )
            return -1;

        // INIT
        xn::DepthMetaData depthMD;
        g_DepthGenerator.GetMetaData( depthMD );

        return cvMatFromXnDepthMetaData( depthMD, pCvDepth8, pCvDepth16 );
    }

    // image to CV::MAT
    int cvMatFromXnImageMetaData( const xn::ImageMetaData &md, cv::Mat *pImgRGB )
    {
        pImgRGB->create( md.FullYRes(), md.FullXRes(), CV_8UC3 );

        // copy
        const XnUInt8* pImageRow = md.Data();
        for ( XnUInt y = 0; y < md.YRes(); ++y, pImageRow += md.XRes() * 3 )
        {
            memcpy( pImgRGB->data + pImgRGB->step * y, pImageRow, md.XRes() * 3 );
        }

        cvtColor( *pImgRGB, *pImgRGB, CV_BGR2RGB );

        return 0;
    }

    // image to CV::MAT
    int nextImageAsMat( xn::ImageGenerator& g_ImageGenerator, cv::Mat *pImgRGB )
    {
        // fetch Metadata
        xn::ImageMetaData md;
        g_ImageGenerator.GetMetaData( md );

        // forward call
        return cvMatFromXnImageMetaData( md, pImgRGB );
    }

    // cat file
    void catFile( const std::string& path)
    {
        FILE *f;
        char line[512];
        //const char fpath[128] = SAMPLE_XML_PATH;
        f = fopen( path.c_str(), "r" );
        if ( f != 0 )
        {
            while ( fgets( line, 512, f ) )
            {
                std::cout << line;
            }
        }
        else
            std::cerr << "file does not exist" << std::endl;

        fclose(f);
        f = 0;
    }

    // Is Point inside Mat?
    bool inside( const cv::Point2i& point, const cv::Mat& mat )
    {
        return ( (point.x >= 0)
                 && (point.y >= 0)
                 && (point.x < mat.cols)
                 && (point.y < mat.rows) );
    }

    // Is Rect inside Mat?
    bool inside( const cv::Rect& rect, const cv::Mat& mat )
    {
        return inside(rect.tl(),mat) && inside(rect.br(),mat);
    }

    void putFloatToMat( cv::Mat &frame, float fValue, const cv::Point& origin, int fontFace, double fontScale, const cv::Scalar& colour )
    {
        char str[10];
        sprintf( str, "%.2f", fValue );
        cv::putText( frame, str, origin, fontFace, fontScale, colour );
    }

    void putIntToMat( cv::Mat &frame, int value, const cv::Point& origin, int fontFace, double fontScale, const cv::Scalar& colour )
    {
        char str[10];
        sprintf( str, "%d", value );
        cv::putText( frame, str, origin, fontFace, fontScale, colour );
    }

    void drawHistogram1D( cv::Mat const& hist, const char* title, int fixHeight )
    {
        double maxVal = 0;
        if ( !fixHeight )
            minMaxLoc( hist, 0, &maxVal, 0, 0 );
        else
            maxVal = 255;

        int scale = 1;
        int width = 10;
        cv::Mat histImg = cv::Mat::zeros( maxVal * scale*1.1f, hist.rows*width, CV_8UC3);
        float scale2 = 255.0 / (float)maxVal;

        for( int h = 0; h < hist.rows; ++h )
        {
            uchar binVal = hist.at<uchar>( h );
            int intensity = cvRound( binVal * scale2 );
            cv::rectangle( histImg,
                         cv::Point( (h+1)*width - 1, histImg.rows - intensity*scale),
                         cv::Point( h*width+1, histImg.rows ),
                         cv::Scalar::all(128 + (255-binVal)/2),
                         CV_FILLED );
        }

        cv::namedWindow( title, 1 );
        cv::imshow( title, histImg );
    }

    float calcAngleRad( cv::Point2f const& a, cv::Point2f const& b)
    {
        return acos( (a.x*b.x+a.y*b.y) / (sqrt(a.x*a.x+a.y*a.y) * sqrt(b.x*b.x+b.y*b.y)));
    }

    float calcAngleRad( cv::Point3f const& a, cv::Point3f const& b)
    {
        return acos( (a.x*b.x+a.y*b.y+a.z*b.z) / (sqrt(a.x*a.x+a.y*a.y+a.z*a.z) * sqrt(b.x*b.x+b.y*b.y+b.z*b.z)));
    }

    /*cv::Point2f addPs( cv::Point2f const& a, cv::Point2f const& b)
    {
        return cv::Point2f( a.x+b.x, a.y+b.y);
    }*/

    cv::Point2f divP( cv::Point2f const& a, float const divisor )
    {
        return cv::Point2f( a.x/divisor, a.y/divisor);
    }

    cv::Point2f avgP( cv::Point const& a, cv::Point const& b)
    {
        return cv::Point2f( float(a.x+b.x)/2.0f, float(a.y+b.y)/2.0f );
    }

    cv::Point3f avgP( cv::Point3f const& a, cv::Point3f const& b)
    {
        return cv::Point3f( (a.x+b.x)/2.0f, (a.y+b.y)/2.0f, (a.z+b.z)/2.0f );
    }

    void normalize( cv::Vec2f &v )
    {
        float length = sqrt(v[0]*v[0]+v[1]*v[1]);
        v[0] /= length;
        v[1] /= length;
    }

    float distance2D( cv::Point3f const& a, cv::Point3f const& b )
    {
        return sqrt((a.x-b.x) * (a.x-b.x) + (a.y-b.y) * (a.y-b.y));
    }

    float distance2D( cv::Vec3f const& a, cv::Vec3f const& b )
    {
        return sqrt((a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]));
    }

    float distance2D( cv::Vec3f const& a, cv::Vec2f const& b )
    {
        return sqrt((a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]));
    }

    float distance2D( cv::Vec2f const& a, cv::Vec3f const& b )
    {
        return sqrt((a[0]-b[0]) * (a[0]-b[0]) + (a[1]-b[1]) * (a[1]-b[1]));
    }

    std::string getCvImageType( int type )
    {
        // find type
        int imgTypeInt = type % 8;
        std::string imgTypeString;

        switch (imgTypeInt)
        {
            case 0:
                imgTypeString = "8U";
                break;
            case 1:
                imgTypeString = "8S";
                break;
            case 2:
                imgTypeString = "16U";
                break;
            case 3:
                imgTypeString = "16S";
                break;
            case 4:
                imgTypeString = "32S";
                break;
            case 5:
                imgTypeString = "32F";
                break;
            case 6:
                imgTypeString = "64F";
                break;
            default:
                break;
        }

        // find channel
        int channel = (type/8) + 1;

        std::stringstream out;
        out << "CV_" << imgTypeString << "C" << channel;

        return out.str();
    }

    std::string outputDirectoryNameWithTimestamp( std::string path )
    {
        std::string outputPath;

        time_t rawtime;
        struct tm * timeinfo;
        char buffer [80];

        time (&rawtime);
        timeinfo = localtime (&rawtime);

        strftime ( buffer, 80, "_%Y%m%d_%H%M", timeinfo );

        outputPath = path + std::string( buffer );
        return outputPath;
    }

    void writePNG( std::string const& name, cv::Mat const& img );
}

#if 0
array<array<int,3>,3> a;
for ( int i = 0;  i < a.size();  ++i )
for ( int j = 0;  j < a[0].size();  ++j )
{
    a[i][j] = i == j ? 1 : -1;
}

for ( auto row : a )
{
    for ( auto e : row )
    {
        cout << e;
    }
    cout << endl;
}
#endif
