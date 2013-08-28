project(TriangleTracing)
cmake_minimum_required(VERSION 2.8)

INCLUDE( ../commonLibLocations.cmake )

aux_source_directory(. SRC_LIST1)
aux_source_directory(./src SRC_LIST2)
aux_source_directory(./src/shaders SRC_LIST3)
file (GLOB H_LIST "include/*.h")

INCLUDE_DIRECTORIES( ${MY_INCLUDES} include )

#add_executable( ${PROJECT_NAME} ${H_LIST}  ${SRC_LIST1} ${SRC_LIST2} ${SRC_LIST3} )
add_library( ${PROJECT_NAME} ${H_LIST}  ${SRC_LIST1} ${SRC_LIST2} ${SRC_LIST3} )

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

#COMMAND {CMAKE_PROGRAM} -E copy_if_different src/shaders/*.frag ${CMAKE_BINARY_DIR}/shaders
#message( "cmakebinarydir: " ${CMAKE_BINARY_DIR})

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/src/shaders
        ${CMAKE_BINARY_DIR}/shaders)
        #$<TARGET_FILE_DIR:${PROJECT_NAME}>/shaders)
