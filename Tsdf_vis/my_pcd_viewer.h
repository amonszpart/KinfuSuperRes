#ifndef MY_PCD_VIEWER_H
#define MY_PCD_VIEWER_H

#include <string>

namespace am
{
    class MyPCDViewer
    {
        public:
            MyPCDViewer();

            int
            run( const std::string &file_name );

            //int
            //loadTsdf( std::string const& file_name, pcl::TSDFVolume<float,short>::Ptr tsdf_volume );
    };

} // ns am

#endif // MY_PCD_VIEWER_H

/*
  .PCD v0.7 - Point Cloud Data file format
VERSION 0.7
FIELDS x y z intensity
SIZE 4 4 4 4
TYPE F F F F
COUNT 1 1 1 1
WIDTH 275889
HEIGHT 1
VIEWPOINT 0 0 0 1 0 0 0
POINTS 275889
DATA binary
*/
