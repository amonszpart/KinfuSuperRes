project(SuperRes-NI-1-5)
cmake_minimum_required(VERSION 2.8)

INCLUDE(localcfg.cmake)

# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5 )
    SET( OPENCV_LOC /home/amonszpart/3rdparty/opencv-2.4.5 )
else(TROLL) # Ubul
    SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
endif(TROLL)

set( dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set( EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set( LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

#SET( CMAKE_PREFIX_PATH ${OPENCV_LOC}/localInstall/share/OpenCV )
#SET( CMAKE_MODULE_PATH ${OPENCV_LOC} )
#SET( OpenCV_DIR ${OPENCV_LOC}/localInstall )
#SET(OPENCV_INSTALL_DIR ${OPENCV_LOC}/localInstall )
#SET(OPENCV_LIB_COMPONENTS "core" "imgproc" "highgui" "features2d" )
#INCLUDE( ${OPENCV_LOC}/findOpenCV.cmake )


# PACKAGES
#FIND_PACKAGE( OpenCV REQUIRED )
#message(STATUS "OpenCV_INCLUDE_DIR: [${OpenCV_INCLUDE_DIR}].")
#message(STATUS "OpenCV_INCLUDE_DIRS: [${OpenCV_INCLUDE_DIRS}].")
#message(STATUS "OpenCV_LIBRARIES: [${OpenCV_LIBRARIES}].")
#message(STATUS "OpenCV_LIBRARIES: [${OpenCV_LIBRARIES}].")

#find_path(OPENCV_INCLUDE_DIR opencv2/core/core.hpp
#    ${OPENCV_LOC}/localInstall/include NO_DEFAULT_PATH
# )
#find_library(OPENCV_LIBRARY_DIR
# NAMES opencv_core
# PATHS
# "${OPENCV_LOC}/localInstall/lib" NO_DEFAULT_PATH
# )
#message(STATUS "prefixpath:" ${CMAKE_PREFIX_PATH})
#message(STATUS "Include file detected: [${OPENCV_INCLUDE_DIR}].")
#message(STATUS "Lib file detected: [${OPENCV_LIBRARY_DIR}].")

#add_library(tesseract STATIC IMPORTED)

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

SET( HEADERS_LIST
    src/MaCvImageBroadcaster.h
    src/HomographyCalculator.h
    src/BilateralFiltering.h
    src/amCommon.h
    src/io/Recorder.h
    src/io/CvImageDumper.h
    src/util/XnVUtil.h
    src/util/MaUtil.h
)

# SRC
SET( SRC_LIST
    src/main.cpp
    src/MaCvImageBroadcaster.cpp
    src/HomographyCalculator.cpp
    src/BilateralFiltering.cpp
    src/io/Recorder.cpp
    src/io/CvImageDumper.cpp
    src/util/MaUtil.cpp
)

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
    #${OPENNI_LOC}/Include
    /usr/include/openni/
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
    ${OPENCV_LOC}/localInstall/include/
    )

# EXECUTABLE
add_executable(${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST})

# LINK
TARGET_LINK_LIBRARIES(${PROJECT_NAME}
   # ${OpenCV_LIBS}
    ${PCL_LOC}/lib/libpcl_common.so
    ${PCL_LOC}/lib/libpcl_io.so
    ${PCL_LOC}/lib/libpcl_visualization.so
    ${PCL_LOC}/lib/libpcl_gpu_kinfu.so
    ${PCL_LOC}/lib/libpcl_gpu_containers.so

    #${OPENNI_LOC}/Lib/libOpenNI.so
#    ${PCL_COMMON_LIBRARIES}
#    ${PCL_IO_LIBRARIES}
    #opencv_core
    #opencv_highgui
    #opencv_imgproc
    #pcl_io
    #pcl_common
    #pcl_visualization
    #pcl_io_ply
    OpenNI
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
    ${OPENCV_LOC}/localInstall/lib/libopencv_core.so
    ${OPENCV_LOC}/localInstall/lib/libopencv_imgproc.so
    ${OPENCV_LOC}/localInstall/lib/libopencv_highgui.so
    ${OPENCV_LOC}/localInstall/lib/libopencv_features2d.so
    ${OPENCV_LOC}/localInstall/lib/libopencv_flann.so
    ${OPENCV_LOC}/localInstall/lib/libopencv_calib3d.so
)

