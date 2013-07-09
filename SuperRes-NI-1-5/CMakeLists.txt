project(SuperRes-NI-1-5)
cmake_minimum_required(VERSION 2.8)

INCLUDE(localcfg.cmake)
#add_subdirectory( src/kinfu_large_scale )

# FLAGS
SET( CMAKE_CXX_COMPILER /usr/bin/g++ )
SET( CMAKE_CXX_FLAGS "-std=c++0x" )

# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
    SET(OPENNI_INCLUDE /usr/include/openni)
else(TROLL) # Ubul
    SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/install )
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
)

# LIBRARIES
SET( MY_LIBRARIES
    ${OpenCV_LIBS}
    OpenNI
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
)

IF(USE_PCL)
    SET( MY_INCLUDES
        ${MY_INCLUDES}
        ${PCL_LOC}/include/pcl-1.7
        ${PCL_LOC}/include/pcl-1.7/pcl/visualization
    )

    SET( MY_LIBRARIES
        ${MY_LIBRARIES}
        ${PCL_LOC}/lib/libpcl_common.so
        ${PCL_LOC}/lib/libpcl_io.so
        ${PCL_LOC}/lib/libpcl_visualization.so
        ${PCL_LOC}/lib/libpcl_gpu_kinfu.so
        ${PCL_LOC}/lib/libpcl_gpu_containers.so
    )

endif(USE_PCL)

######################## PROJECT files ##############################

SET( HEADERS_LIST
    src/MaCvImageBroadcaster.h
    src/HomographyCalculator.h
    src/BilateralFiltering.h
    src/amCommon.h
    src/io/Recorder.h
    src/io/CvImageDumper.h
    src/util/XnVUtil.h
    src/util/MaUtil.h
    src/prism_camera_parameters.h
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
    src/yetAnotherRecorder.cpp
)

# INCLUDES
INCLUDE_DIRECTORIES( ${MY_INCLUDES} )

# EXECUTABLE
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${MY_LIBRARIES} )
