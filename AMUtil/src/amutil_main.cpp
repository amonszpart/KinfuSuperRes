#include "AMUtil2.h"

#include "opencv2/core/core.hpp"

#include <iostream>
#include <cstdlib>
#include "limits.h"

int testPFM()
{
    std::cout << "testing savePFM and loadPFM with random data..." << std::endl;
    srand( time(NULL) );

    cv::Mat m( 32, 24, CV_32FC1 );
    for ( int y = 0; y < m.rows; ++y )
    {
        for ( int x = 0; x < m.cols; ++x )
        {
            m.at<float>( y, x ) = (float)rand() / RAND_MAX;
        }
    }

    am::util::savePFM( m, "floatm.pfm" );

    cv::Mat mread;
    am::util::loadPFM( mread, "floatm.pfm" );

    float err = 0.f;
    for ( int y = 0; y < mread.rows; ++y )
    {
        for ( int x = 0; x < mread.cols; ++x )
        {
            err += fabs(mread.at<float>(y,x) - m.at<float>(y,x));
        }
    }
    float avgerr = err/m.rows/m.cols;
    std::cout << "avg err: " << avgerr << std::endl;

    std::cout << "test " << ((avgerr < 0.01) ? "PASSED" : "FAILED") << std::endl;
    return 0;
}

int main( int argv, char** argc )
{
    std::cout << "hello amutil2" << std::endl;
    int res = 0;
    res |= testPFM();

    return res;
}
