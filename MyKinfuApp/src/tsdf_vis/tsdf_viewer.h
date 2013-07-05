#ifndef TSDF_VIEWER_H
#define TSDF_VIEWER_H

#include <pcl/gpu/kinfu/kinfu.h>
#include "../kinfu/tools/tsdf_volume.h"
#include "../kinfu/tools/tsdf_volume.hpp"

namespace am
{
    class TSDFViewer
    {
        public:
            TSDFViewer();
            pcl::gpu::KinfuTracker kinfu_;
            pcl::TSDFVolume<float, short> tsdf_volume_;

            void
            loadTsdfFromFile( std::string path, bool binary );
    };

} // ns am

#endif // TSDF_VIEWER_H
