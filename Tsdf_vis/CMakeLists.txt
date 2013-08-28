cmake_minimum_required(VERSION 2.8)
project(Tsdf_vis)

INCLUDE( ../commonLibLocations.cmake )

MESSAGE(STATUS "PCL_LOC: " ${PCL_LOC})
MESSAGE(STATUS "MYINCLUDES: " ${MY_INCLUDES})
MESSAGE(STATUS "MYLIBS: " ${MY_LIBRARIES})
MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})

### BilateralFilteringCuda
ADD_LIBRARY( imp_bilFilCuda STATIC IMPORTED)
SET_PROPERTY( TARGET imp_bilFilCuda PROPERTY
              IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/build/libcuBilateralFilteringCuda.a )

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/include/ )
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../BilateralFilteringCuda/inc/ )

### MyKinfuTracker

add_library ( libMyKinfuTracker STATIC IMPORTED)
set_property( TARGET libMyKinfuTracker PROPERTY
              IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/build/libMyKinfuTracker.a )
SET(KINFU_LIB libMyKinfuTracker)

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/include/ )
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/src/ )

### MyKinfuApp

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuApp/src/util )
INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuApp/src/kinfu/tools )

### TriangleTracing

add_library ( libTriangleTracing STATIC IMPORTED)
set_property( TARGET libTriangleTracing PROPERTY
              IMPORTED_LOCATION ${CMAKE_CURRENT_SOURCE_DIR}/../TriangleTracing/build/libTriangleTracing.a )

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../TriangleTracing/include/ )
#INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../MyKinfuTracker/src/ )


####################

SET( GNU_FLAGS ${GNU_FLAGS} "-O3 -Wno-deprecated" )

SET( TSDF_VIS_NAME "tsdf_vis" )

SET( TSDF_VIS_HEADER_LIST
    ../MyKinfuApp/src/kinfu/tools/tsdf_volume.h
    ../MyKinfuApp/src/kinfu/tools/tsdf_volume.hpp
    ../MyKinfuApp/src/kinfu/tools/kinfu_util.h
    ../MyKinfuApp/src/kinfu/tools/my_screenshot_manager.h
    ../MyKinfuApp/src/util/MaUtil
    include/DepthViewer3D.h
    include/MeshRayCaster.h
    include/tsdf_viewer.h
    include/YangFilteringWrapper.h
    include/AmPclUtil.h
    include/ProcessorWithIntrinsics.h
    include/PolyMeshViewer.h
    include/UpScaling.h
    include/vtkCatmullClarkFilter.h
)

SET( TSDF_VIS_SRC_LIST
    ../MyKinfuApp/src/kinfu/tools/kinfu_util.cpp
    ../MyKinfuApp/src/kinfu/tools/my_screenshot_manager.cpp
    src/DepthViewer3D.cpp
    src/MeshRayCaster.cpp
    src/tsdf_viewer.cpp
    src/YangFilteringWrapper.cpp
    src/AmPclUtil.cpp
    ../MyKinfuApp/src/util/MaUtil
    src/PolyMeshViewer.cpp
    src/UpScaling.cpp
    src/vtkCatmullClarkFilter.cpp
    main.cpp
)

# INCLUDES
INCLUDE_DIRECTORIES( include ${MY_INCLUDES} ${KINFU_INCLUDE} )

# EXECUTABLE
ADD_EXECUTABLE( ${TSDF_VIS_NAME} ${TSDF_VIS_SRC_LIST} ${TSDF_VIS_HEADER_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${TSDF_VIS_NAME}
    libTriangleTracing
    GL
    GLEW
    glut
    GLU
    imp_bilFilCuda
    ${KINFU_LIB}
    ${MY_LIBRARIES}
    ${PCL_LOC}/lib/libpcl_octree.so
    ${PCL_LOC}/lib/libpcl_surface.so
    vtkGraphics
)
