project(MyKinfuApp)
cmake_minimum_required(VERSION 2.8)

INCLUDE(localcfg.cmake)
#add_subdirectory( src/kinfu_large_scale )

SET( CMAKE_CXX_COMPILER /usr/bin/g++)

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

# HEADERS
SET( KINFU_HEADERS_LIST
    src/kinfu/tools/kinfu_pcl_headers.h
    src/kinfu/tools/evaluation.h
    #src/kinfu/tools/openni_capture.h
    src/kinfu/tools/tsdf_volume.h
    src/kinfu/tools/tsdf_volume.hpp
    src/kinfu/tools/scene_cloud_view.h
    src/kinfu/tools/point_cloud_color_handler_rgb_cloud.hpp
    src/kinfu/tools/image_view.h
    src/kinfu/tools/kinfu_util.h
    src/kinfu/tools/my_screenshot_manager.h
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
              #${KINFU_LS_SRC_LIST}
)
message(STATUS "KINFU_LS_SRC_LIST: " ${KINFU_LS_SRC_LIST})

# FLAGS
SET(CMAKE_CXX_FLAGS "-std=c++0x")

# INCLUDES
include_directories(

	${PCL_LOC}/include/pcl-1.7
    #${PCL_LOC}/include/pcl-1.7/io
	${PCL_LOC}/include/pcl-1.7/pcl/visualization

    /usr/include/eigen3/

    ${OPENNI_INCLUDE}

    /usr/include/vtk-5.8/

    ${CUDA_PATH}/include
    )

# EXECUTABLE
add_executable(${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST} )

# LINK
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
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

message( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
