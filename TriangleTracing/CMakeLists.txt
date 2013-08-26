project(TriangleTracing)
cmake_minimum_required(VERSION 2.8)

INCLUDE( ../commonLibLocations.cmake )

aux_source_directory(. SRC_LIST1)
aux_source_directory(./src SRC_LIST2)
aux_source_directory(./src/shaders SRC_LIST3)

INCLUDE_DIRECTORIES( ${MY_INCLUDES} )

add_executable( ${PROJECT_NAME} ${SRC_LIST1} ${SRC_LIST2} ${SRC_LIST3})

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

