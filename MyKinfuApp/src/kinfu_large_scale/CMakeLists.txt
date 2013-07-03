cmake_minimum_required(VERSION 2.8)

#SET(HEADERS_LIST
#)

message(STATUS "CMAKE_CURRENT_SOURCE_DIR: " ${CMAKE_CURRENT_SOURCE_DIR})

SET( KINFU_LS_SRC_LIST
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/kinfuLS_app.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/process_kinfu_large_scale_output.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/record_maps_rgb.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/kinfu_app_sim.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/capture.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/standalone_texture_mapping
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/record_tsdfvolume
    ${CMAKE_CURRENT_SOURCE_DIR}/tools/record_maps_rgb
)

add_library( kinfu_large_scale ${KINFU_LS_SRC_LIST} )

