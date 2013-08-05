#include "FreenectPlayer.h"

FreenectPlayer::FreenectPlayer()
{
}

#include "libfreenect.hpp"
#include <iostream>
#include <vector>
#include <cmath>
#include <pthread.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>


using namespace cv;
using namespace std;

class Mutex {
public:
    Mutex() {
        pthread_mutex_init( &m_mutex, NULL );
    }
    void lock() {
        pthread_mutex_lock( &m_mutex );
    }
    void unlock() {
        pthread_mutex_unlock( &m_mutex );
    }
private:
    pthread_mutex_t m_mutex;
};

class MyFreenectDevice : public Freenect::FreenectDevice {
  public:
    MyFreenectDevice(freenect_context *_ctx, int _index)
        : Freenect::FreenectDevice(_ctx, _index), m_buffer_depth(FREENECT_DEPTH_11BIT),m_buffer_rgb(FREENECT_VIDEO_RGB), m_gamma(2048), m_new_rgb_frame(false), m_new_depth_frame(false),
          depthMat(Size(640,480),CV_16UC1), rgbMat(Size(640,480),CV_8UC3,Scalar(0)), ownMat(Size(640,480),CV_8UC3,Scalar(0))
    {
        for( unsigned int i = 0 ; i < 2048 ; i++) {
            float v = i/2048.0;
            v = std::pow(v, 3)* 6;
            m_gamma[i] = v*6*256;
        }
    }
    // Do not call directly even in child
    void VideoCallback(void* _rgb, uint32_t timestamp) {
        //std::cout << "RGB callback" << std::endl;
        m_rgb_mutex.lock();
        uint8_t* rgb = static_cast<uint8_t*>(_rgb);
        rgbMat.data = rgb;
        m_new_rgb_frame = true;
        m_rgb_mutex.unlock();
    };
    // Do not call directly even in child
    void DepthCallback(void* _depth, uint32_t timestamp) {
        //std::cout << "Depth callback" << std::endl;
        m_depth_mutex.lock();
        uint16_t* depth = static_cast<uint16_t*>(_depth);
        depthMat.data = (uchar*) depth;
        m_new_depth_frame = true;
        m_depth_mutex.unlock();
    }

    bool getVideo(Mat& output) {
        m_rgb_mutex.lock();
        if(m_new_rgb_frame) {
            cv::cvtColor(rgbMat, output, CV_RGB2BGR);
            m_new_rgb_frame = false;
            m_rgb_mutex.unlock();
            return true;
        } else {
            m_rgb_mutex.unlock();
            return false;
        }
    }

    bool getDepth(Mat& output) {
            m_depth_mutex.lock();
            if(m_new_depth_frame) {
                depthMat.copyTo(output);
                m_new_depth_frame = false;
                m_depth_mutex.unlock();
                return true;
            } else {
                m_depth_mutex.unlock();
                return false;
            }
        }

  private:
    std::vector<uint8_t> m_buffer_depth;
    std::vector<uint8_t> m_buffer_rgb;
    std::vector<uint16_t> m_gamma;
    Mat depthMat;
    Mat rgbMat;
    Mat ownMat;
    Mutex m_rgb_mutex;
    Mutex m_depth_mutex;
    bool m_new_rgb_frame;
    bool m_new_depth_frame;
};



int main(int argc, char **argv)
{
    bool die(false);
    string rgbFilePrefix("img8_");
    string grayFilePrefix("gray8_");
    string depFilePrefix("dep11_");
    string rgbSuffix(".jpg");
    string depSuffix(".pgm");
    int i_snap(0),iter(0);

    Mat depthMat(Size(640,480),CV_16UC1);
    Mat depthf  (Size(640,480),CV_8UC1);
    Mat rgbMat(Size(640,480),CV_8UC3,Scalar(0));
    Mat ownMat(Size(640,480),CV_8UC3,Scalar(0));

        //The next two lines must be changed as Freenect::Freenect isn't a template but the method createDevice:
        //Freenect::Freenect<MyFreenectDevice> freenect;
        //MyFreenectDevice& device = freenect.createDevice(0);
        //by these two lines:
        Freenect::Freenect freenect;
        MyFreenectDevice& device = freenect.createDevice<MyFreenectDevice>(0);

    namedWindow("rgb",CV_WINDOW_AUTOSIZE);
    namedWindow("depth",CV_WINDOW_AUTOSIZE);
    device.startVideo();
    device.startDepth();
    while (!die) {
        device.getVideo(rgbMat);
        device.getDepth(depthMat);
        cv::imshow("rgb", rgbMat);
        depthMat.convertTo(depthf, CV_8UC1, 255.0/2048.0);
        cv::imshow("depth",depthf);
        char k = cvWaitKey(5);
        if( k == 27 ){
            cvDestroyWindow("rgb");
            cvDestroyWindow("depth");
            break;
        }
        if( k == 32 )
        {
            //std::ostringstream rgbFileName;
            //rgbFileName << rgbFilePrefix << i_snap << rgbSuffix;
            char rgbFileName[1024];
            sprintf( rgbFileName,"%s%08d%s", rgbFilePrefix.c_str(), i_snap, rgbSuffix.c_str() );
            std::cout << "dumping to " << rgbFileName << std::endl;
            cv::imwrite( rgbFileName, rgbMat, std::vector<int>{ CV_IMWRITE_JPEG_QUALITY,100} );

            sprintf( rgbFileName,"%s%08d%s", rgbFilePrefix.c_str(), i_snap, ".pgm" );
            std::cout << "dumping to " << rgbFileName << std::endl;
            cv::imwrite( rgbFileName, rgbMat );

            cv::Mat rgbGray;
            cv::cvtColor( rgbMat, rgbGray, CV_RGB2GRAY );
            sprintf( rgbFileName,"%s%08d%s", grayFilePrefix.c_str(), i_snap, ".pgm" );
            std::cout << "dumping to " << rgbFileName << std::endl;
            cv::imwrite( rgbFileName, rgbGray );

            //std::ostringstream depFileName;
            //depFileName << depFilePrefix << i_snap << depSuffix;
            char depFileName[1024];
            sprintf( depFileName,"%s%08d%s", depFilePrefix.c_str(), i_snap, depSuffix.c_str() );
            std::cout << "dumping to " << depFileName << std::endl;
            cv::imwrite( depFileName, depthMat, std::vector<int>{ CV_IMWRITE_PNG_COMPRESSION,0} );

            i_snap++;
        }
        //if(iter >= 1000) break;
        iter++;
    }
    std::cout << "stopping..." << std::endl;

    device.stopVideo();
    device.stopDepth();
    return 0;
}
