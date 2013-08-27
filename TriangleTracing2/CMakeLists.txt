project(TriangleTracing2)
cmake_minimum_required(VERSION 2.8)

INCLUDE( ../commonLibLocations.cmake )

file (GLOB H_LIST "include/*.h")
message("My headers: " ${H_LIST})
INCLUDE_DIRECTORIES( include
                     ${MY_INCLUDES} )

aux_source_directory( ./src SRC_LIST )

add_executable( ${PROJECT_NAME} ${H_LIST} ${SRC_LIST} )

TARGET_LINK_LIBRARIES( ${PROJECT_NAME}
    GL
    GLEW
    glut
    GLU
    opencv_core
    opencv_highgui
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
