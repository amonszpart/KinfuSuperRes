project(MyKinfuApp)
cmake_minimum_required(VERSION 2.8)

INCLUDE(../localcfg.cmake)
#add_subdirectory( src/kinfu_large_scale )

INCLUDE( ../commonLibLocations.cmake )
SET( MYKINFU TRUE )

#############################################################################################

#add_subdirectory( src/calibration )

############## MY_KINFU_APP ##############

####################
### BilateralFilteringCuda
add_library ( imp_bilFilCuda SHARED IMPORTED)
set_property( TARGET imp_bilFilCuda PROPERTY
              IMPORTED_LOCATION_RELEASE ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/build/${CMAKE_BUILD_TYPE}/libBilateralFilteringCuda.so )
set_property( TARGET imp_bilFilCuda PROPERTY
              IMPORTED_LOCATION_DEBUG ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/build/${CMAKE_BUILD_TYPE}/libBilateralFilteringCudad.so )

INCLUDE_DIRECTORIES( "../BilateralFilteringCuda/inc/" )
INCLUDE_DIRECTORIES( "../BilateralFilteringCuda/include/" )

MESSAGE(STATUS "!!!!!!": ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/build/${CMAKE_BUILD_TYPE}/libBilateralFilteringCuda.so )

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
add_definitions( -DPCL_BUILT_WITH_VTK ) # needed by amImageGrabber
### MA
SET( HEADERS_LIST
    ${KINFU_TRACKER_Hs}
    ${KINFU_HEADERS_LIST}
    src/util/MaUtil.h
    src/util/XnVUtil.h
    src/myopenni/MyONIGrabber.h
    src/myopenni/MyOpenNIGrabber.h
    src/kinfu/amImageGrabber.h
    #src/kinfu/AMKinfuTracker.h
    ../BilateralFilteringCuda/include/ViewPointMapperCuda.h
    ../BilateralFilteringCuda/include/BilateralFilterCuda.hpp
)

SET( SRC_LIST
    ${KINFU_TRACKER_CPPs}
    ${KINFU_SRC_LIST}
    src/util/MaUtil.cpp
    src/myopenni/MyONIGrabber.cpp
    src/myopenni/MyOpenNIGrabber.cpp
    src/kinfu/amImageGrabber.cpp
    src/main.cpp
    #src/kinfu/AMKinfuTracker.cpp
)

### INCLUDES
IF(MYKINFU)
    add_definitions(-DMYKINFU)
    SET( KINFU_INCLUDE "${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/include/" )
ELSE(MYKINFU)
    SET( KINFU_INCLUDE ${PCL_LOC}/include/pcl-1.7/pcl/gpu/kinfu/ )
ENDIF(MYKINFU)

INCLUDE_DIRECTORIES( ${KINFU_INCLUDE} ${MY_INCLUDES} )

### EXECUTABLE
ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST} )

### LINK

IF(MYKINFU)
    SET( KINFU_LIB libMyKinfuTracker )
ELSE(MYKINFU)
    SET( KINFU_LIB ${PCL_LOC}/lib/libpcl_gpu_kinfu.so )
ENDIF(MYKINFU)

message(STATUS MY_LIBRARIES: ${MY_LIBRARIES})


TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
    imp_bilFilCuda
    ${KINFU_LIB}
    ${MY_LIBRARIES}
)

############## TSDF_VIS ##############

#add_subdirectory( src/tsdf_vis )
