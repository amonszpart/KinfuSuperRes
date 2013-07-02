#include "BilateralFiltering.h"
#include <math.h>

#include <opencv2/imgproc/imgproc.hpp>

namespace am
{
    BilateralFiltering::BilateralFiltering( double sigmaD, double sigmaR )
    {
        const int    sigmaMax         = std::max( sigmaD, sigmaR );
        _kernelRadius = ceil( 2.0 * sigmaMax );

        const double twoSigmaRSquared = 2.0 * sigmaR * sigmaR;
        const int    kernelSize       = _kernelRadius * 2 + 1;
        _kernelD      = cv::Mat( kernelSize, kernelSize, CV_64FC1 );

        for ( int x = 0; x < kernelSize; ++x )
        {
            for ( int y = 0; y < kernelSize; ++y )
            {
                _kernelD.at<double>( y, x ) = this->gauss( sigmaD, x - _kernelRadius, y - _kernelRadius );
            }
        }

        for ( unsigned i = 0U; i < _gaussSimilarity.size(); ++i )
        {
            _gaussSimilarity[i] = exp( -(static_cast<double>(i) / twoSigmaRSquared) );
        }
    }

    BilateralFiltering::~BilateralFiltering()
    {
    }

    double BilateralFiltering::getSpatialWeight( int m, int n, int i, int j ) const
    {
        return _kernelD.at<double>( (int)(j-n + _kernelRadius), (int)(i-m + _kernelRadius) );
    }

    double BilateralFiltering::similarity( int p, int s ) const
    {
        // this equals: Math.exp(-(( Math.abs(p-s)) /  2 * this->sigmaR * this->sigmaR));
        // but is precomputed to improve performance
        return this->_gaussSimilarity[ abs(p-s) ];
    }

    double BilateralFiltering::gauss( double sigma, int x, int y ) const
    {
        return exp(-((x * x + y * y) / (2.0 * sigma * sigma)));
    }

    /*bool BilateralFiltering::isInsideBoundaries( cv::Mat const& img, int y, int x ) const
    {
        return (
                    (y > -1)          &&
                    (x > -1)          &&
                    (y < img.rows) &&
                    (x < img.cols)
                    );
    }*/

} // namespace
