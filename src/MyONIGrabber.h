#ifndef MYONIGRABBER_H
#define MYONIGRABBER_H

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/common/time.h>

class SimpleONIProcessor
{
    public:

        bool save;

        void cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud);
        void imageDepthImageCallback (const boost::shared_ptr<openni_wrapper::Image>& c_img, const boost::shared_ptr<openni_wrapper::DepthImage>& d_img, float constant );
        void run ();
};


#endif // MYONIGRABBER_H
