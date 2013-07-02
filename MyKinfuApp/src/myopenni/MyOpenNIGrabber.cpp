#include "MyOpenNIGrabber.h"

#include "../util/MaUtil.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

MyOpenNIGrabber::MyOpenNIGrabber()
{
}

void
MyOpenNIGrabber::cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
{
    static unsigned count = 0;
    static double last = pcl::getTime ();
    if (++count == 30)
    {
        double now = pcl::getTime ();
        std::cout << "distance of center pixel :" << cloud->points [(cloud->width >> 1) * (cloud->height + 1)].z << " mm. Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
        count = 0;
        last = now;
    }

    if (save)
    {
        std::stringstream ss;
        ss << std::setprecision(12) << pcl::getTime () * 100 << ".pcd";
        pcl::io::savePCDFile (ss.str (), *cloud);
        std::cout << "wrote point clouds to file " << ss.str() << std::endl;
    }
}

void
MyOpenNIGrabber::imageDepthImageCallback (const boost::shared_ptr<openni_wrapper::Image>& c_img, const boost::shared_ptr<openni_wrapper::DepthImage>& d_img, float constant )
{
    static unsigned count = 0;
    static double last = pcl::getTime ();
    if (++count == 30)
    {
        double now = pcl::getTime ();
        std::cout << "got synchronized image x depth-image with constant factor: " << constant << ". Average framerate: " << double(count)/double(now - last) << " Hz" <<  std::endl;
        std::cout << "Depth baseline: " << d_img->getBaseline () << " and focal length: " << d_img->getFocalLength () << std::endl;
        count = 0;
        last = now;
    }

    std::cout << "c_img->getEncoding(): " << c_img->getEncoding() << std::endl;
    std::cout << "d_img size: " << d_img->getWidth() << "x" << d_img->getHeight() << std::endl;

    cv::Mat cvImg;
    util::cvMatFromXnImageMetaData( c_img->getMetaData(), &cvImg );
    cv::imshow("cvImg", cvImg );
    cv::waitKey(5);
}

void MyOpenNIGrabber::run ()
{
    save = false;

    // create a new grabber for OpenNI devices
    pcl::Grabber* interface = new pcl::OpenNIGrabber();

    // make callback function from member function
    boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
            boost::bind (&MyOpenNIGrabber::cloud_cb_, this, _1);

    // connect callback function for desired signal. In this case its a point cloud with color values
    boost::signals2::connection c = interface->registerCallback (f);

    // make callback function from member function
    boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant)> f2 =
            boost::bind (&MyOpenNIGrabber::imageDepthImageCallback, this, _1, _2, _3);

    // connect callback function for desired signal. In this case its a point cloud with color values
    boost::signals2::connection c2 = interface->registerCallback (f2);

    // start receiving point clouds
    interface->start ();

    std::cout << "<Esc>, \'q\', \'Q\': quit the program" << std::endl;
    std::cout << "\' \': pause" << std::endl;
    std::cout << "\'s\': save" << std::endl;
    char key;
    do
    {
        key = static_cast<char> (getchar ());
        switch (key)
        {
            case ' ':
                if (interface->isRunning())
                    interface->stop();
                else
                    interface->start();
            case 's':
                save = !save;
        }
    } while (key != 27 && key != 'q' && key != 'Q');

    // stop the grabber
    interface->stop ();
}
