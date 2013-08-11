#pragma once
#ifndef __MY_KINFU_PCL_HEADERS
#define __MY_KINFU_PCL_HEADERS

#include <pcl/console/parse.h>

#include <boost/filesystem.hpp>

// from MyKinfuTracker/include
#include "kinfu.h"
#include "raycaster.h"
#include "marching_cubes.h"

//#include <pcl/gpu/kinfu/kinfu.h>
//#include <pcl/gpu/kinfu/raycaster.h>
//#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/gpu/containers/initialization.h>

#include <pcl/common/time.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/image_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_io.h>
//#include <pcl/io/png_io.h>
#include <pcl/io/openni_grabber.h>
#include <pcl/io/oni_grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/exceptions.h>

#include <pcl/visualization/point_cloud_color_handlers.h>

#ifdef HAVE_OPENCV
#   include <opencv2/highgui/highgui.hpp>
#   include <opencv2/imgproc/imgproc.hpp>
#   include <pcl/gpu/utils/timers_opencv.hpp>

/*       */ typedef pcl::gpu::ScopeTimerCV ScopeTimeT;
#else
/*       */ typedef pcl::ScopeTime ScopeTimeT;
#endif

namespace nsKinFuApp
{
        enum { PCD_BIN = 1, PCD_ASCII = 2, PLY = 3, MESH_PLY = 7, MESH_VTK = 8 };
};

#endif // __MY_KINFU_PCL_HEADERS
