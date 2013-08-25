#ifndef AM_UTIL_H
#define AM_UTIL_H
#pragma once

/* ----------------------------------------
 * INCLUDES
 * ---------------------------------------- */

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <string>

/* ----------------------------------------
 * DEFINES
 * ---------------------------------------- */

#ifndef UCHAR
#define UCHAR
typedef unsigned char uchar;
#endif

#include "XnVUtil.h"

/*#define CHECK_RC(rc, what)											\
    if (rc != XN_STATUS_OK)											\
{																\
    printf("%s failed: %s\n", what, xnGetStatusString(rc));		\
    return rc;													\
    }

#define CHECK_ERRORS(rc, errors, what)		\
    if (rc == XN_STATUS_NO_NODE_PRESENT)	\
{										\
    XnChar strError[1024];				\
    errors.ToString(strError, 1024);	\
    printf("%s\n", strError);			\
    return (rc);						\
    }
*/

#define scUC(a) static_cast<unsigned char>(a)

/* ----------------------------------------
 * TEMPLATES
 * ---------------------------------------- */

template <class T>
unsigned char MAXINDEX3( T a, T b, T c )
{
    unsigned char ret = 0;
    if ( a > b )
        if ( a > c )
            ret = 0;
        else
            ret = 2;
    else
        if ( b > c )
            ret = 1;
        else
            ret = 2;
    return ret;
}

template <class T>
inline bool between( T a, T lower, T upper )
{
    return (a <= upper) && (a > lower);
}

/* ----------------------------------------
 * PREDECLARATIONS
 * ---------------------------------------- */

namespace xn
{
    class DepthGenerator;
    class ImageGenerator;
    class ImageMetaData;
    class DepthMetaData;
}


/* ----------------------------------------
 * METHODS
 * ---------------------------------------- */

namespace util
{

    unsigned int getClosestPowerOfTwo( unsigned int n );

    int cvMatFromXnDepthMetaData( const xn::DepthMetaData &md, cv::Mat* pCvDepth8, cv::Mat* pCvDepth16 );
    int nextDepthToMats( xn::DepthGenerator& g_DepthGenerator, cv::Mat* pCvDepth8, cv::Mat* pCvDepth16 );

    int cvMatFromXnImageMetaData( const xn::ImageMetaData &md, cv::Mat *pImgRGB );
    int nextImageAsMat( xn::ImageGenerator& g_ImageGenerator, cv::Mat *pImgRGB );



    void catFile( const std::string& path );

    inline double NEXTDOUBLE() { return (static_cast<double>(rand()) / static_cast<double>(RAND_MAX)); };
    inline double NEXTFLOAT()  { return (static_cast<float> (rand()) / static_cast<float> (RAND_MAX)); };

    bool inside( const cv::Point2i& point, const cv::Mat& mat );
    bool inside( const cv::Rect& rect, const cv::Mat& mat );

    void putFloatToMat( cv::Mat &frame, float fValue, const cv::Point& origin, int fontFace, double fontScale, const cv::Scalar& colour );
    void putIntToMat( cv::Mat &frame, int value, const cv::Point& origin, int fontFace, double fontScale, const cv::Scalar& colour );
    void drawHistogram1D( cv::Mat const& hist, const char* title, int fixHeight = 0 );

    float calcAngleRad( cv::Point2f const& a, cv::Point2f const& b);
    float calcAngleRad( cv::Point3f const& a, cv::Point3f const& b);

    //inline cv::Point2f addPs( cv::Point2f const& a, cv::Point2f const& b);
    cv::Point2f divP( cv::Point2f const& a, float const divisor );
    cv::Point2f avgP( cv::Point const& a, cv::Point const& b);
    cv::Point3f avgP( cv::Point3f const& a, cv::Point3f const& b);
    void normalize( cv::Vec2f &v );
    float distance2D( cv::Point3f const& a, cv::Point3f const& b );
    float distance2D( cv::Vec3f const& a, cv::Vec3f const& b );
    float distance2D( cv::Vec3f const& a, cv::Vec2f const& b );
    float distance2D( cv::Vec2f const& a, cv::Vec3f const& b );

    std::string getCvImageType( int type );

    extern
    std::string outputDirectoryNameWithTimestamp( std::string path );

    extern
    void writePNG( std::string const& name, cv::Mat const& img );
} // end ns util

namespace am
{
    namespace util
    {
        void blend( cv::Mat &blendedUC3, cv::Mat const& dep, float depMax, cv::Mat const& rgb, float rgbMax = 255.f );
    } // end ns util
} // end ns am

#endif // UTIL_H

