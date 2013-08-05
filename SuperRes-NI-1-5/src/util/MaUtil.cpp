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

    // depth to CV::MAT
    int nextDepthToMats( xn::DepthGenerator& g_DepthGenerator, cv::Mat* pCvDepth8, cv::Mat* pCvDepth16 )
    {
        static constexpr double conversionRatio = 255.0/2047.0;

        // CHECK
        if ( (pCvDepth8 == nullptr) && (pCvDepth16 == nullptr) )
            return -1;

        // INIT
        xn::DepthMetaData depthMD;
        cv::Mat cvDepth16;
        g_DepthGenerator.GetMetaData( depthMD );
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

    // image to CV::MAT
    int nextImageAsMat( xn::ImageGenerator& g_ImageGenerator, cv::Mat *pImgRGB )
    {
        // update
        xn::ImageMetaData g_imageMD;
        cv::Mat imgBGR;

        // INIT
        g_ImageGenerator.GetMetaData( g_imageMD );
        imgBGR.create( g_imageMD.FullYRes(), g_imageMD.FullXRes(), CV_8UC3 );

        // copy
        const XnUInt8* pImageRow = g_imageMD.Data();
        for ( XnUInt y = 0; y < g_imageMD.YRes(); ++y, pImageRow += g_imageMD.XRes() * 3 )
        {
            memcpy( imgBGR.data + imgBGR.step*y, pImageRow, g_imageMD.XRes() * 3 );
        }

        // OUT
        if ( pImgRGB != nullptr )
        {
            pImgRGB->create( imgBGR.size(), imgBGR.type() );
            cvtColor( imgBGR, *pImgRGB, CV_BGR2RGB );
        }

        return 0;
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

    cv::Vec3b blend( cv::Vec3b rgb, ushort dep, float alpha, ushort maxDepth, uchar maxColor)
    {
        cv::Vec3f ret =
                (
                    alpha          * (cv::Vec3f)rgb                                    / (float)maxColor
                    +
                    (1.f - alpha) * ((cv::Vec3f){(float)dep, (float)dep, (float)dep}) / (float)maxDepth
                    ) * maxColor;

        return cv::Vec3b( round(ret[0]), round(ret[1]), round(ret[2]) );
    }

    cv::Vec3b blend( ushort dep, cv::Vec3b rgb, float alpha, ushort maxDepth, uchar maxColor)
    {
        return blend( rgb, dep, 1.f - alpha, maxDepth, maxColor );
    }

    uchar blend( uchar dep, uchar rgb, float alpha, ushort maxDepth, uchar maxColor)
    {
        float ret =
                (
                    alpha          * (float)rgb  / (float)maxColor
                    +
                    (1.f - alpha) * ((float)dep) / (float)maxDepth
                    ) * maxColor;

        return uchar(round(ret));
    }


    std::string printBool( bool b )
    {
        return b ? "ON" : "OFF";
    }

    /*void overlayImage1OntoImage3( cv::Mat &img1, cv::Mat &img2, float alpha )
    {
        cv::Mat tmp1, *p1, *p2;
        p1 = &img1;
        p2 = &img2;
        if ( img1.size != img2.size )
        {
            cv::resize( img1, tmp1, img2.size(), 0, 0, cv::INTER_CUBIC );
            p1 = &tmp1;
        }

        cv::Mat_<uchar>::const_iterator itEnd = p1->end<T1>();
        uint y = 0, x = 0;
        for ( cv::Mat_<uchar>::const_iterator it = p1->begin<uchar>(); it != itEnd; it++ )
        {
            // read
            T1 val1 = p1->at<uchar>( y, x );
            T2 val2 = p2->at<uchar>( y, x );
            if ( val1 )
            {
                if ( val2 )
                {
                    p2->at<T2>( y, x ) = blend( val1, val2, alpha );
                }
                else
                {
                    p2->at<T2>( y, x ) = val1;
                }
            }
            else
            {
                p2->at<T2>( y, x ) = val2;
            }

            // iterate coords
            if ( ++x == static_cast<uint>(p1->cols) )
            {
                x = 0;
                ++y;
            }
        }
    }*/


    /*
     * @brief Copies content of cvImg to img.
     * @param cvImg Holds grayscale, 0..1 float data
     * @param img Initialized 2D double array
     */
    void CopyCvImgToDouble( cv::Mat const& cvImg, double**& img )
    {
        if ( img == NULL )
        {
            std::cerr << "cvImgToDouble: double** img is uninitalized, please call qx_allocd()..." << std::endl;
            return;
        }

        if ( cvImg.type() != CV_32FC1 )
        {
            std::cerr << "cvImgToDouble: cvImg needs to be " << std::endl;
            return;
        }

        for ( int y = 0; y < cvImg.rows; ++y )
            for ( int x = 0; x < cvImg.cols; ++x )
        {
                img[y][x] = cvImg.at<float>(y,x);
        }
    }

    /*
     * @brief       Copies content of img to cvImg.
     * @param img   Initialized 2D double array
     * @param h     Height of img
     * @param w     Width of img
     * @param cvImg Holds grayscale, 0..1 float data
     */
    void CopyDoubleToCvImage( double** const& img, unsigned h, unsigned w, cv::Mat & cvImg )
    {
        if ( img == NULL )
        {
            std::cerr << "CopyDoubleToCvImage: double** img is uninitalized, please call qx_allocd()..." << std::endl;
            return;
        }

        cvImg.create( h, w, CV_32FC1 );

        for ( int y = 0; y < cvImg.rows; ++y )
            for ( int x = 0; x < cvImg.cols; ++x )
        {
                cvImg.at<float>( y, x ) = img[y][x];
        }
    }

} // ns util

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
