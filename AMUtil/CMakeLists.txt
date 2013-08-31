cmake_minimum_required(VERSION 2.8)
SET(PROJECT_NAME AMUtil)
project( ${PROJECT_NAME} )

#INCLUDE( ../commonLibLocations.cmake )

# INCLUDES
INCLUDE_DIRECTORIES( include )

SET( SRC_LIST
    src/AMUtil2.cpp
    src/amutil_main.cpp
)

# EXECUTABLE
ADD_LIBRARY( ${PROJECT_NAME} ${SRC_LIST} )
#ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
    opencv_highgui
    opencv_core
)
