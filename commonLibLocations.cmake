INCLUDE( ../localcfg.cmake )
# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
    SET(OPENNI_INCLUDE /usr/include/openni)
else(TROLL) # Ubul
    SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/install )
    #SET( PCL_LOC /usr/ )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
    SET(OPENNI_INCLUDE /usr/include/ni)
endif(TROLL)

# PACKAGES
FIND_PACKAGE( OpenCV REQUIRED )

#FIND_PACKAGE( PCL 1.3 REQUIRED COMPONENTS common io )
#link_directories(${PCL_LIBRARY_DIRS})

# FLAGS
SET( CMAKE_CXX_COMPILER /usr/bin/g++ )
#SET( CMAKE_CXX_FLAGS "-std=c++0x -O3 -Wno-deprecated" )
SET( CMAKE_CXX_FLAGS "-std=c++0x" )
#SET( GNU_FLAGS ${GNU_FLAGS} "-O3 -Wno-deprecated")

# DEFS
add_definitions( -D__x86_64__=1 -Dlinux=1 -DHAVE_OPENCV=1 -DHAVE_OPENNI=1 )
#add_definitions( ${PCL_DEFINITIONS} )

# BUILD
MESSAGE(STATUS "Setting caching build dir to: ${CMAKE_CURRENT_SOURCE_DIR}/build" )
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
    /usr/lib/libOpenNI.so
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
    vtkImaging
    vtkIO
    ${CUDA_PATH}/lib64/libcudart.so
)
