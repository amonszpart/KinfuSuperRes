#ifndef BILATERALFILTERING_H
#define BILATERALFILTERING_H

#include <opencv2/core/core.hpp>
#include <array>
#include "amCommon.h"
#include <iostream>

namespace am
{

    using std::cout;
    using std::endl;

    enum MISSING_DATA { IGNORE, USE }; // IGNORE: skip, if 0, USE: use everything

    class BilateralFiltering
    {
        protected:
            int                    _kernelRadius;
            cv::Mat                _kernelD;
            std::array<double,256> _gaussSimilarity;

            //bool   isInsideBoundaries( const cv::Mat &img, int m, int n ) const;
            double similarity        ( int p,int s )                      const;
            double gauss             ( double sigma, int x, int y )       const;
            double getSpatialWeight  ( int m, int n,int i,int j )         const;

        public:
            BilateralFiltering( double sigmaD, double sigmaR );
            virtual ~BilateralFiltering();

            // Run in channel
            template <typename T>
            int runFilter( cv::Mat const& img, cv::Mat &out, MISSING_DATA bMissMode = IGNORE ) const
            {
                cout << "starting BF" << endl;

                out = cv::Mat::zeros( img.rows, img.cols, img.type() );

                for ( int y = 0; y < img.rows; ++y )
                {
                    for ( int x = 0; x < img.cols; ++x )
                    {
                        for ( uchar c = 0; c < img.channels(); ++c )
                        {
                            double sum              = 0.0;
                            double totalWeight      = 0.0;
                            int    intensityCenter  = img.at<T>( y, img.channels() * x + c );

                            int mMax = std::min( y + _kernelRadius, img.rows );
                            int nMax = std::min( x + _kernelRadius, img.cols );
                            double weight;

                            for ( int m = std::max( y -_kernelRadius, 0 ); m < mMax; ++m )
                            {
                                for ( int n = std::max( x-_kernelRadius, 0 ); n < nMax; ++n )
                                {
                                    // read neighbour
                                    T intensityKernelPos = img.at<T>( m, img.channels() * n + c );

                                    // skip, if zero
                                    if ( (bMissMode == IGNORE) && (intensityKernelPos == 0) )
                                         continue;

                                    // include
                                    weight      = getSpatialWeight( m, n, y, x ) * similarity( intensityKernelPos, intensityCenter );
                                    totalWeight += weight;
                                    sum         += ( weight * intensityKernelPos );
                                }
                            }

                            out.at<T>( y, out.channels() * x + c ) = floor( sum / totalWeight );
                        }
                    }
                }

                return AM_SUCCESS;
            }

            // run cross channel
            template <typename depT, typename imgT>
            int runFilter( cv::Mat const& dep, cv::Mat const& img, cv::Mat &out, MISSING_DATA bMissMode = IGNORE ) const
            {
                using std::vector;
                //const int C_Size = 3; // how deep is C
                //const int L = C_Size * C_Size;

                cout << "starting CrossBilateralFilter" << endl;

                // validate input
                CV_Assert( (img.cols == dep.cols) && (img.rows == dep.rows) && dep.channels() == 1 );

                // init output
                out = cv::Mat::zeros( dep.rows, dep.cols, dep.type() );

                // init Cost
                //vector<cv::Mat> C;
                //for ( unsigned i = 0; i < C_Size; ++i )
                //    C.push_back( cv::Mat::zeros(dep.rows, dep.cols, CV_32SC1) );

                imgT intensityCenter[ img.channels() ];
                for ( int y = 0; y < img.rows; ++y )
                {
                    for ( int x = 0; x < img.cols; ++x )
                    {
                        double sum              = 0.0;
                        double totalWeight      = 0.0;

                        int mMax = std::min( y + _kernelRadius, img.rows );
                        int nMax = std::min( x + _kernelRadius, img.cols );
                        double weight, simWeight;
                        for ( uchar c = 0; c < img.channels(); ++c )
                            intensityCenter[c] = img.at<imgT>( y, img.channels() * x + c );

                        for ( int m = std::max( y -_kernelRadius, 0 ); m < mMax; ++m )
                        {
                            for ( int n = std::max( x - _kernelRadius, 0 ); n < nMax; ++n )
                            {
                                // skip, if zero
                                if ( (bMissMode == IGNORE) && (dep.at<depT>(m,n) == 0) )
                                    continue;

                                simWeight = 0.0;
                                for ( uchar c = 0; c < img.channels(); ++c )
                                {
                                    simWeight += similarity( img.at<imgT>( m, img.channels() * n + c ), intensityCenter[c] );
                                }
                                weight      = getSpatialWeight( m, n, y, x ) * 0.33 * simWeight;
                                totalWeight += weight;
                                sum         += weight * dep.at<depT>( m, n );
                            }
                        }

                        out.at<depT>( y, x ) = round( sum / totalWeight );
                    }
                }

                return AM_SUCCESS;
            }
    };
}

#endif // BILATERALFILTERING_H
