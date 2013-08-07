#include <pcl/console/parse.h>

#include "util/MaUtil.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "myopenni/MyONIGrabber.h"

namespace am
{
    int mainKinfuApp ( int argc, char* argv[] );
}

int printUsage()
{
    std::cout << "\nUsage: \n\tMyKinfuApp -oni input_path.oni -out cloudName\n" << std::endl;
    return 1;
}

int
main (int argc, char* argv[] )
{
    if (pcl::console::find_argument (argc, argv, "-h" ) > 0)
    {
        return printUsage ();
    }

    if (pcl::console::find_argument (argc, argv, "-oni" ) < 0)
    {
        std::cout << "Please provide an -oni file argument..." << std::endl;
        printUsage ();
        return 1;
    }
    if (pcl::console::find_argument (argc, argv, "-out" ) < 0)
    {
        std::cout << "Please provide an -out file argument..." << std::endl;
        printUsage ();
        return 1;
    }

#if 1
    //--viz 0 -oni /home/amonszpart/cpp_projects/SuperRes-NI-1-5/build/out/imgs_20130701_1440/recording_push.oni
    //--viz 0 -oni /home/amonszpart/cpp_projects/SuperRes-NI-1-5/build/out/imgs_20130701_1627/recording_push.oni
    // -oni ~/rec/troll_recordings/keyboard_imgs_20130701_1440/recording_push.oni -out ~/rec/testing/keyboard_cross_nomap -ic --eval ~/rec/testing/eval --viz 1
    std::cout << "running main.cpp" << std::endl;
    return am::mainKinfuApp( argc, argv );
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

