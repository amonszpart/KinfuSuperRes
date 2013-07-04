project(MyKinfuApp)
cmake_minimum_required(VERSION 2.8)

INCLUDE(localcfg.cmake)
#add_subdirectory( src/kinfu_large_scale )

# FLAGS
SET( CMAKE_CXX_COMPILER /usr/bin/g++ )
SET( CMAKE_CXX_FLAGS "-std=c++0x" )

# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
    SET(OPENNI_INCLUDE /usr/include/openni)
else(TROLL) # Ubul
    SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
    SET(OPENNI_INCLUDE /usr/include/ni)
endif(TROLL)

# PACKAGES
FIND_PACKAGE( OpenCV REQUIRED )

#FIND_PACKAGE( PCL 1.3 REQUIRED COMPONENTS common io )
#link_directories(${PCL_LIBRARY_DIRS})

# DEFS
add_definitions( -D__x86_64__=1 -Dlinux=1 -DHAVE_OPENCV=1 -DHAVE_OPENNI=1 )
#add_definitions( ${PCL_DEFINITIONS} )

# BUILD
set( dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set( EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set( LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

# INCLUDES
SET( MY_INCLUDES
    /usr/include/eigen3/
    /usr/include/vtk-5.8/
    ${CUDA_PATH}/include
    ${OPENNI_INCLUDE}
    ${PCL_LOC}/include/pcl-1.7
    ${PCL_LOC}/include/pcl-1.7/pcl/visualization
)

# LIBRARIES
SET( MY_LIBRARIES
    ${OpenCV_LIBS}
    ${PCL_LOC}/lib/libpcl_common.so
    ${PCL_LOC}/lib/libpcl_io.so
    ${PCL_LOC}/lib/libpcl_visualization.so
    ${PCL_LOC}/lib/libpcl_gpu_kinfu.so
    ${PCL_LOC}/lib/libpcl_gpu_containers.so
    OpenNI
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
)

#############################################################################################


############## MY_KINFU_APP ##############

# HEADERS
SET( KINFU_HEADERS_LIST
    src/kinfu/tools/kinfu_pcl_headers.h
    src/kinfu/tools/evaluation.h
    src/kinfu/tools/tsdf_volume.h
    src/kinfu/tools/tsdf_volume.hpp
    src/kinfu/tools/scene_cloud_view.h
    src/kinfu/tools/point_cloud_color_handler_rgb_cloud.hpp
    src/kinfu/tools/image_view.h
    src/kinfu/tools/kinfu_util.h
    src/kinfu/tools/my_screenshot_manager.h
    #src/kinfu/tools/openni_capture.h
)
SET( HEADERS_LIST
    src/util/MaUtil.h
    src/util/XnVUtil.h
    src/myopenni/MyONIGrabber.h
    src/myopenni/MyOpenNIGrabber.h
    ${KINFU_HEADERS_LIST}
)

# SRC
SET( KINFU_SRC_LIST
    src/kinfu/tools/kinfu_app.cpp
    #src/kinfu/tools/capture.cpp
    src/kinfu/tools/evaluation.cpp
    #src/kinfu/tools/record_tsdfvolume.cpp
    src/kinfu/tools/scene_cloud_view.cpp
    src/kinfu/tools/image_view.cpp
    src/kinfu/tools/kinfu_util.cpp
    src/kinfu/tools/my_screenshot_manager.cpp
)

SET( SRC_LIST src/main.cpp
              src/util/MaUtil.cpp
              src/myopenni/MyONIGrabber.cpp
              src/myopenni/MyOpenNIGrabber.cpp
              ${KINFU_SRC_LIST}
)

# INCLUDES
INCLUDE_DIRECTORIES( ${MY_INCLUDES} )

# EXECUTABLE
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${MY_LIBRARIES} )

############## TSDF_VIS ##############

SET( TSDF_VIS_NAME "TSDFVis" )

SET( HEADER_LIST

)

SET( SRC_LIST
    src/tsdf_vis/main.cpp
)

# INCLUDES
INCLUDE_DIRECTORIES( ${MY_INCLUDES} )

# EXECUTABLE
ADD_EXECUTABLE( ${TSDF_VIS_NAME} ${SRC_LIST} ${HEADERS_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${TSDF_VIS_NAME} ${MY_LIBRARIES} )
