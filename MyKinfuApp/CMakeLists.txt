project(MyKinfuApp)
cmake_minimum_required(VERSION 2.8)

INCLUDE(../localcfg.cmake)
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
    ${PCL_LOC}/include/pcl-1.7
    ${PCL_LOC}/include/pcl-1.7/pcl/visualization
    #${PCL_LOC}/include/pcl-1.7/pcl/gpu/kinfu/
    #src/kinfu/include/pcl/gpu/kinfu/
)

# LIBRARIES
SET( MY_LIBRARIES
    ${OpenCV_LIBS}
    ${PCL_LOC}/lib/libpcl_common.so
    ${PCL_LOC}/lib/libpcl_io.so
    ${PCL_LOC}/lib/libpcl_visualization.so
    ${PCL_LOC}/lib/libpcl_gpu_containers.so
    OpenNI
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
    ${CUDA_PATH}/lib64/libcudart.so
)

#############################################################################################

#add_subdirectory( src/calibration )

############## MY_KINFU_APP ##############

####################
### BilateralFilteringCuda
add_library ( imp_bilFilCuda STATIC IMPORTED)
set_property( TARGET imp_bilFilCuda PROPERTY
              IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/build/libcuBilateralFilteringCuda.a )

INCLUDE_DIRECTORIES( "../BilateralFilteringCuda/inc/" )
INCLUDE_DIRECTORIES( "../BilateralFilteringCuda/src/" )

####################

####################
### MyKinfuTracker

add_library ( libMyKinfuTracker STATIC IMPORTED)
set_property( TARGET libMyKinfuTracker PROPERTY
              IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/build/libMyKinfuTracker.a )

#INCLUDE_DIRECTORIES( "../MyKinfuTracker/include/" )
#INCLUDE_DIRECTORIES( "../BilateralFilteringCuda/src/" )

####################

#SET( KINFU_TRACKER_Hs
#    src/kinfu/include/pcl/gpu/kinfu/color_volume.h
#    src/kinfu/include/pcl/gpu/kinfu/tsdf_volume.h
#    src/kinfu/include/pcl/gpu/kinfu/marching_cubes.h
#    src/kinfu/include/pcl/gpu/kinfu/pixel_rgb.h
#    src/kinfu/include/pcl/gpu/kinfu/raycaster.h
#    src/kinfu/include/pcl/gpu/kinfu/kinfu.h
#    src/kinfu/src/safe_call.hpp
#)
#SET( KINFU_TRACKER_CPPs
#    src/kinfu/src/color_volume.cpp
#    src/kinfu/src/tsdf_volume.cpp
#    src/kinfu/src/marching_cubes.cpp
#    src/kinfu/src/raycaster.cpp
#    src/kinfu/src/kinfu.cpp
#)

####################


### HEADERS
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
    #src/kinfu/src/internal.h
    #src/kinfu/tools/openni_capture.h
)

### SRC
SET( KINFU_SRC_LIST
    src/kinfu/tools/kinfu_app.cpp
    #src/kinfu/tools/capture.cpp
    src/kinfu/tools/evaluation.cpp
    #src/kinfu/tools/record_tsdfvolume.cpp
    src/kinfu/tools/scene_cloud_view.cpp
    src/kinfu/tools/image_view.cpp
    src/kinfu/tools/kinfu_util.cpp
    src/kinfu/tools/my_screenshot_manager.cpp
    src/kinfu/mainKinfuApp.cpp
)

### MA
SET( HEADERS_LIST
    ${KINFU_TRACKER_Hs}
    ${KINFU_HEADERS_LIST}
    src/util/MaUtil.h
    src/util/XnVUtil.h
    src/myopenni/MyONIGrabber.h
    src/myopenni/MyOpenNIGrabber.h
    #src/kinfu/AMKinfuTracker.h
    ../BilateralFilteringCuda/src/ViewPointMapperCuda.h
    ../BilateralFilteringCuda/src/BilateralFilterCuda.hpp
)

SET( SRC_LIST
    ${KINFU_TRACKER_CPPs}
    ${KINFU_SRC_LIST}
    src/main.cpp
    src/util/MaUtil.cpp
    src/myopenni/MyONIGrabber.cpp
    src/myopenni/MyOpenNIGrabber.cpp
    #src/kinfu/AMKinfuTracker.cpp
)

### INCLUDES
SET(MYKINFU TRUE)

IF(MYKINFU)
    add_definitions(-DMYKINFU)
    SET( KINFU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/include/" )
    #INCLUDE_DIRECTORIES( "../MyKinfuTracker/include/" )
ELSE(MYKINFU)
    SET( KINFU_INCLUDE ${PCL_LOC}/include/pcl-1.7/pcl/gpu/kinfu/ )
    #INCLUDE_DIRECTORIES( ${PCL_LOC}/include/pcl-1.7/pcl/gpu/kinfu/ )
ENDIF(MYKINFU)

INCLUDE_DIRECTORIES( ${KINFU_INCLUDE} ${MY_INCLUDES} )

### EXECUTABLE
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST} )

### LINK
TARGET_LINK_LIBRARIES( ${PROJECT_NAME} imp_bilFilCuda )

IF(MYKINFU)
    SET( KINFU_LIB libMyKinfuTracker )
    #TARGET_LINK_LIBRARIES( ${PROJECT_NAME} libMyKinfuTracker )
ELSE(MYKINFU)
    SET( KINFU_LIB ${PCL_LOC}/lib/libpcl_gpu_kinfu.so )
    #TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${PCL_LOC}/lib/libpcl_gpu_kinfu.so )
ENDIF(MYKINFU)

TARGET_LINK_LIBRARIES( ${PROJECT_NAME} ${KINFU_LIB} ${MY_LIBRARIES} )

############## TSDF_VIS ##############

add_subdirectory( src/tsdf_vis )
