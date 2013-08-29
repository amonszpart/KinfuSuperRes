project(TriangleTracing)
cmake_minimum_required(VERSION 2.8)

INCLUDE( ../commonLibLocations.cmake )

#aux_source_directory(./src SRC_LIST )
aux_source_directory(./src/shaders SHADERS_LIST )
file (GLOB H_LIST "include/*.h")

SET( SRC_LIST
        src/math_3d.cpp
        src/mesh.cpp
        src/textfile.cpp
        src/TriangleRenderer.cpp
        src/main.cpp
)

INCLUDE_DIRECTORIES( ${MY_INCLUDES} include )

#add_executable( ${PROJECT_NAME} ${H_LIST} ${SRC_LIST} ${SHADERS_LIST} )
add_library( ${PROJECT_NAME} ${H_LIST} ${SRC_LIST} ${SHADERS_LIST} )

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
    GL
    GLEW
    glut
    GLU
    opencv_core
    opencv_highgui
    ${PCL_LOC}/lib/libpcl_common.so
    ${PCL_LOC}/lib/libpcl_io.so
    #${PCL_LOC}/lib/libpcl_visualization.so
    #${PCL_LOC}/lib/libpcl_gpu_containers.so
    boost_system
    boost_filesystem
    boost_thread
    #vtkCommon
    #vtkFiltering
    #vtkRendering
    #vtkImaging
    #vtkIO
)

#add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
#    COMMAND ${CMAKE_COMMAND} -E copy_directory
#        ${CMAKE_SOURCE_DIR}/src/shaders
#        ${CMAKE_BINARY_DIR}/shaders)
