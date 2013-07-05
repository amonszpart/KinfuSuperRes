#include "tsdf_viewer.h"

namespace am
{

    TSDFViewer::TSDFViewer()
    {

    }

    void
    TSDFViewer::loadTsdfFromFile( std::string path, bool binary )
    {
        tsdf_volume_.load( path, binary );
        //kinfu_.volume() = tsdf_volume_;
        //kinfu_.volume().load( path, binary );
    }

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


} // ns am
