/*
 * Software License Agreement (BSD License)
 *
 *  Copyright (c) 2011, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * Author: Suat Gedikli (gedikli@willowgarage.com)
 */

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/common/time.h>
#include <pcl/console/parse.h>

#include "util/MaUtil.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "MyONIGrabber.h"

class SimpleOpenNIProcessor
{
    public:

        bool save;

        void
        cloud_cb_ (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr &cloud)
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
        imageDepthImageCallback (const boost::shared_ptr<openni_wrapper::Image>& c_img, const boost::shared_ptr<openni_wrapper::DepthImage>& d_img, float constant )
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

        void run ()
        {
            save = false;

            // create a new grabber for OpenNI devices
            pcl::Grabber* interface = new pcl::OpenNIGrabber();

            // make callback function from member function
            boost::function<void (const pcl::PointCloud<pcl::PointXYZRGBA>::ConstPtr&)> f =
                    boost::bind (&SimpleOpenNIProcessor::cloud_cb_, this, _1);

            // connect callback function for desired signal. In this case its a point cloud with color values
            boost::signals2::connection c = interface->registerCallback (f);

            // make callback function from member function
            boost::function<void (const boost::shared_ptr<openni_wrapper::Image>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant)> f2 =
                    boost::bind (&SimpleOpenNIProcessor::imageDepthImageCallback, this, _1, _2, _3);

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
};

int mainKinfuApp ( int argc, char* argv[] );

int
main (int argc, char* argv[] )
{
#if 1
    //--viz 0 -oni /home/amonszpart/cpp_projects/SuperRes-NI-1-5/build/out/imgs_20130701_1440/recording_push.oni
    //--viz 0 -oni /home/amonszpart/cpp_projects/SuperRes-NI-1-5/build/out/imgs_20130701_1627/recording_push.oni
    std::cout << "running main.cpp" << std::endl;
    return mainKinfuApp( argc, argv );
#endif
    std::string oni_file;
    boost::shared_ptr<pcl::Grabber> capture;

    if (pcl::console::parse_argument (argc, argv, "-oni", oni_file) > 0)
    {
      std::cout << "oni: " << oni_file << std::endl;

      SimpleONIProcessor v;
      v.run ( oni_file );

      std::cout << "finished reading oni..." << std::endl;
    }


    return 0;
}

