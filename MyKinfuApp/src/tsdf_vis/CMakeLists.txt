cmake_minimum_required(VERSION 2.8)

MESSAGE(STATUS "MYINCLUDES: " ${MY_INCLUDES})
MESSAGE(STATUS "MYLIBS: " ${MY_LIBRARIES})
MESSAGE(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES( ${CMAKE_CURRENT_SOURCE_DIR}/../../MyKinfuTracker/include/ )

SET( TSDF_VIS_NAME "tsdf_vis" )

SET( TSDF_VIS_HEADER_LIST
    ../kinfu/tools/tsdf_volume.h
    ../kinfu/tools/tsdf_volume.hpp
    ../kinfu/tools/kinfu_util.h
    tsdf_viewer.h
    my_pcd_viewer.h
)

SET( TSDF_VIS_SRC_LIST
    ../kinfu/tools/kinfu_util.cpp
    tsdf_viewer.cpp
    my_pcd_viewer.cpp
    main.cpp
)

# INCLUDES
INCLUDE_DIRECTORIES( ${KINFU_INCLUDE} ${MY_INCLUDES} )

# EXECUTABLE
ADD_EXECUTABLE( ${TSDF_VIS_NAME} ${TSDF_VIS_SRC_LIST} ${TSDF_VIS_HEADER_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${TSDF_VIS_NAME}
    ${KINFU_LIB}
    ${MY_LIBRARIES}
)
