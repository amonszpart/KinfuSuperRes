#include "MyThrustUtil.h"
#include <thrust/device_vector.h>

#include "AmCudaUtil.h"

#include <opencv2/core/core.hpp>

MyThrustUtil::MyThrustUtil()
{
}


extern int testKernel();
int MyThrustUtil::runTest()
{
    testKernel();
}

// min(sqr(A - x),truncAt)
extern int squareDiffScalarKernel( float *a, int size, float x, float* out, float truncAt );
int MyThrustUtil::squareDiff( cv::Mat const& a, float x, cv::Mat &out, float truncAt )
{
    if ( a.type() != CV_32FC1 )
    {
        std::cerr << "MyThrustUtil::squareDiff expects CV_32FC1...exiting..." << std::endl;
        return 1;
    }

    // copy input
    float *pA = NULL;
    toContinuousFloat( a, pA );

    // prepare output
    float* pOut = new float[ a.cols * a.rows ];

    // work
    squareDiffScalarKernel( pA, a.cols * a.rows, x, pOut, truncAt );

    // copy output
    out.create( a.size(), CV_32FC1 );
    fromContinuousFloat( pOut, out );

    delete [] pA;
    delete [] pOut;

    return 0;
}

// min(sqr(A - B),truncAt)
int squareDiffKernel( float *a, int size, float* b, float *out, float truncAt );
int MyThrustUtil::squareDiff( cv::Mat const& a, cv::Mat const& b, cv::Mat & out, float truncAt )
{
    if ( a.type() != CV_32FC1 || b.type() != CV_32FC1 )
    {
        std::cerr << "MyThrustUtil::squareDiff expects CV_32FC1...exiting..." << std::endl;
        return 1;
    }

    // copy input
    float *pA = NULL;
    toContinuousFloat( a, pA );
    float *pB = NULL;
    toContinuousFloat( a, pB );

    // prepare output
    float* pOut = new float[ a.cols * a.rows ];

    // work
    squareDiffKernel( pA, a.cols * a.rows, pB, pOut, truncAt );

    // copy output
    out.create( a.size(), CV_32FC1 );
    fromContinuousFloat( pOut, out );

    if ( pA   ) delete [] pA;
    if ( pB   ) delete [] pB;
    if ( pOut ) delete [] pOut;

    return 0;
}

/*
 * @brief performs "minDs( find(C < minC) ) = d"
 */
extern int minMaskedCopyKernel(
        float*  const& Cprev ,
        float*  const& C     ,
        float          d     ,
        int            size  ,
        float*       & minC  ,
        float*       & minDs ,
        float*       & minCm1,
        float*       & minCp1 );
int MyThrustUtil::minMaskedCopy( const cv::Mat &Cprev ,
                                 const cv::Mat &C     ,
                                 const float   d      ,
                                       cv::Mat &minC  ,
                                       cv::Mat &minDs ,
                                       cv::Mat &minCm1,
                                       cv::Mat &minCp1 )
{
    if ( C.type() != CV_32FC1 || minC.type() != CV_32FC1 || minDs.type() != CV_32FC1 )
    {
        std::cerr << "MyThrustUtil::minMaskedCopy expects CV_32FC1...exiting..." << std::endl;
        return 1;
    }

    // copy input
    float *pCprev = NULL;
    toContinuousFloat( Cprev, pCprev );
    float *pC = NULL;
    toContinuousFloat( C, pC );
    float *pMinC = NULL;
    toContinuousFloat( minC, pMinC );
    float *pMinDs = NULL;
    toContinuousFloat( minDs, pMinDs );
    float *pMinCm1 = NULL;
    toContinuousFloat( minCm1, pMinCm1 );
    float *pMinCp1 = NULL;
    toContinuousFloat( minCp1, pMinCp1 );

    // work
    minMaskedCopyKernel( pCprev, pC, d, C.cols * C.rows, pMinC, pMinDs, pMinCm1, pMinCp1 );

    // copy output
    fromContinuousFloat( pMinC , minC );
    fromContinuousFloat( pMinDs , minDs );
    fromContinuousFloat( pMinCm1, minCm1 );
    fromContinuousFloat( pMinCp1, minCp1 );

    if ( pCprev  ) delete [] pCprev;
    if ( pC      ) delete [] pC;
    if ( pMinC   ) delete [] pMinC;
    if ( pMinDs  ) delete [] pMinDs;
    if ( pMinCm1 ) delete [] pMinCm1;
    if ( pMinCp1 ) delete [] pMinCp1;

    return 0;
}

/*
 * @brief   performs per pixel subpixel refinement
 */
extern int subpixelRefineKernel( float*  const& minC  ,
                                 float*  const& minCm1,
                                 float*  const& minCp1,
                                 int            size  ,
                                 float*       & minDs  );
int MyThrustUtil::subpixelRefine( cv::Mat const& minC  ,
                                  cv::Mat const& minCm1,
                                  cv::Mat const& minCp1,
                                  cv::Mat &minDs )
{
    // copy input
    float *pMinC = NULL;
    toContinuousFloat( minC, pMinC );
    float *pMinDs = NULL;
    toContinuousFloat( minDs, pMinDs );
    float *pMinCm1 = NULL;
    toContinuousFloat( minCm1, pMinCm1 );
    float *pMinCp1 = NULL;
    toContinuousFloat( minCp1, pMinCp1 );

    // work
    subpixelRefineKernel( pMinC, pMinCm1, pMinCp1, minC.cols * minC.rows, pMinDs );

    // copy output
    fromContinuousFloat( pMinDs , minDs );

    if ( pMinC   ) delete [] pMinC;
    if ( pMinCm1 ) delete [] pMinCm1;
    if ( pMinCp1 ) delete [] pMinCp1;
    if ( pMinDs  ) delete [] pMinDs;

    return 0;
}
