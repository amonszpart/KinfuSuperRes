project(PclTest)
cmake_minimum_required(VERSION 2.8)

INCLUDE(localcfg.cmake)

# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
else(TROLL) # Ubul
    SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
endif(TROLL)

# PACKAGES
FIND_PACKAGE( OpenCV REQUIRED )

#FIND_PACKAGE( PCL 1.3 REQUIRED COMPONENTS common io )
#link_directories(${PCL_LIBRARY_DIRS})

# DEFS
add_definitions( -D__x86_64__=1 -Dlinux=1 -DHAVE_OPENCV=1 )
#add_definitions( ${PCL_DEFINITIONS} )

# BUILD
set( dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set( EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set( LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

# HEADERS
SET( KINFU_HEADERS_LIST
    src/kinfu/tools/evaluation.h
    src/kinfu/tools/openni_capture.h
    src/kinfu/tools/tsdf_volume.h
    src/kinfu/tools/tsdf_volume.hpp
)
SET( HEADERS_LIST
    src/util/MaUtil.h
    src/util/XnVUtil.h
    src/MyONIGrabber.h
    ${KINFU_HEADERS_LIST}
)

# SRC
SET( KINFU_SRC_LIST
    src/kinfu/tools/kinfu_app.cpp
    #src/kinfu/tools/capture.cpp
    src/kinfu/tools/evaluation.cpp
    #src/kinfu/tools/record_tsdfvolume.cpp
)

SET( SRC_LIST src/main.cpp
              src/util/MaUtil.cpp
              src/MyONIGrabber.cpp
              ${KINFU_SRC_LIST}
)

#src/kinfu/src/kinfu.cpp
#src/kinfu/src/color_volume.cpp

#aux_source_directory(src/kinfu SRC_LIST)

# FLAGS
SET(CMAKE_CXX_FLAGS "-std=c++0x")

# INCLUDES
include_directories(
	${PCL_LOC}/include/pcl-1.7
	${PCL_LOC}/include/pcl-1.7/io
	${PCL_LOC}/include/pcl-1.7/pcl/visualization
    #/usr/local/include/pcl-1.6/
    #/usr/local/include/pcl-1.6/io/
    #/usr/local/include/pcl-1.6/pcl/visualization
    /usr/include/eigen3/
    ${OPENNI_LOC}/Include
    /usr/include/vtk-5.8/
    #${PCL_TRUNK}/include
    #${PCL_TRUNK}/common/include
    #${PCL_TRUNK}/build/include
    #${PCL_TRUNK}/io/include
    #${PCL_TRUNK}/visualization/include
    #${PCL_TRUNK}/geometry/include
    #${PCL_TRUNK}/search/include
    #${PCL_TRUNK}/gpu/kinfu/include
    #${PCL_TRUNK}/gpu/containers/include
    #${PCL_TRUNK}/gpu/utils/include
    ${CUDA_PATH}/include
    )

# EXECUTABLE
add_executable(${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST})

# LINK
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${PCL_LOC}/lib/libpcl_common.so
    ${PCL_LOC}/lib/libpcl_io.so
    ${PCL_LOC}/lib/libpcl_visualization.so
    ${PCL_LOC}/lib/libpcl_gpu_kinfu.so
    ${PCL_LOC}/lib/libpcl_gpu_containers.so

    ${OPENNI_LOC}/Lib/libOpenNI.so
#    ${PCL_COMMON_LIBRARIES}
#    ${PCL_IO_LIBRARIES}
    #opencv_core
    #opencv_highgui
    #opencv_imgproc
    #pcl_io
    #pcl_common
    #pcl_visualization
    #pcl_io_ply
    #OpenNI
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
)

