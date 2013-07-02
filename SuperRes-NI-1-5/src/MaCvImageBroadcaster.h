#ifndef CVIMAGEBROADCASTER_H
#define CVIMAGEBROADCASTER_H

#include <opencv2/core/core.hpp>

#include <list>
#include <memory>

namespace cvTracking
{
    class MaCvImageListener
    {
        public:
            virtual void update( const cv::Mat& /* depthImage*/){};
            virtual void update( const cv::Mat& rgbImage, const cv::Mat& depthImage ){};

            virtual void sendKey( unsigned char key ){};
    };

    class MaCvImageBroadcaster
    {
            std::list<std::shared_ptr<MaCvImageListener> > listeners;

        public:
            int addListener( std::shared_ptr<MaCvImageListener> newListener );

            void update( const cv::Mat& newImage );
            void update( const cv::Mat& rgbImage, const cv::Mat& depthImage );
            void sendKey( unsigned char key );
    };

} // ns cvTracking

#endif // CVIMAGEBROADCASTER_H
