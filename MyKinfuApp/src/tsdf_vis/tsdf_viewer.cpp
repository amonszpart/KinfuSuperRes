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


} // ns am
