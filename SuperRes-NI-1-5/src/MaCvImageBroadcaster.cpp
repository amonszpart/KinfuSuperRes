#include "MaCvImageBroadcaster.h"

namespace cvTracking
{

    void MaCvImageBroadcaster::update( const cv::Mat& depthImage )
    {
        for ( auto it = listeners.begin(); it != listeners.end(); ++it )
        {
            (*it)->update( depthImage );
        }
    }

    void MaCvImageBroadcaster::update( const cv::Mat& rgbImage, const cv::Mat& depthImage )
    {
        for ( auto it = listeners.begin(); it != listeners.end(); ++it )
        {
            (*it)->update( rgbImage, depthImage );
        }
    }

    int MaCvImageBroadcaster::addListener( std::shared_ptr<MaCvImageListener> newListener )
    {
        listeners.push_back( newListener );
        return listeners.size();
    }

    void MaCvImageBroadcaster::sendKey( unsigned char key )
    {
        for ( auto it = listeners.begin(); it != listeners.end(); ++it )
        {
            (*it)->sendKey( key );
        }
    }

} // ns cvTracking
