INCLUDE( ../localcfg.cmake )

# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
    SET(OPENNI_INCLUDE /usr/include/openni)
else(TROLL) # Ubul
    SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk-gpu/install )
    #SET( PCL_LOC /usr/ )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
    SET(OPENNI_INCLUDE /usr/include/ni)
endif(TROLL)

### new
set( EIGEN_DIR  "/usr/local/include/eigen3/unsupported/")
SET( OpenCV_DIR "/home/bontius/workspace/3rdparty/opencv-trunk/install/share/OpenCV" )
SET( PCL_DIR    "/home/bontius/workspace/3rdparty/pcl-trunk/install/share/pcl-1.7/")
SET( OPENNI_INCLUDE_DIRS "/usr/include/ni" )

#
#INCLUDE_DIRECTORIES( ${EIGEN_DIR} )
FIND_PACKAGE( OpenGL REQUIRED)
FIND_PACKAGE( Qt4 REQUIRED )
FIND_PACKAGE( OpenCV COMPONENTS imgproc highgui core contrib REQUIRED)
FIND_PACKAGE( PCL REQUIRED )
FIND_PACKAGE( CUDA REQUIRED )
#FIND_PACKAGE(glut REQUIRED)

ADD_DEFINITIONS( -D__x86_64__=1
                 -Dlinux=1
                 -DHAVE_OPENCV=1
                 -DHAVE_OPENNI=1
                 -DMULTI_LABEL=1
                 -DQT_NO_KEYWORDS
                 -DBOOST_DISABLE_ASSERTS
                 -DBOOST_ENABLE_ASSERT_HANDLER )
###

SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk-gpu/install )
SET( CUDA_PATH /usr/lib/nvidia-319-updates)

# PACKAGES
#link_directories(${PCL_LIBRARY_DIRS})

# FLAGS
SET( CMAKE_CXX_COMPILER /usr/bin/g++ )
#SET( CMAKE_CXX_FLAGS "-std=c++0x -O3 -Wno-deprecated" )
SET( CMAKE_CXX_FLAGS "-std=c++0x" )
SET( CMAKE_CXX_FLAGS_DEBUG "-g" )
SET( CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-deprecated" )
#SET( GNU_FLAGS ${GNU_FLAGS} "-O3 -Wno-deprecated")

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
    ${OPENNI_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    #${GLUT_INCLUDE_DIRS}
    #${OPENGL_INCLUDE_DIRS}
    #/usr/lib/x86_64-linux-gnu/ #cuda?

    #${PCL_LOC}/include/pcl-1.7
    #${PCL_LOC}/include/pcl-1.7/pcl/visualization
    #${PCL_LOC}/include/pcl-1.7/pcl/gpu/kinfu/
    #src/kinfu/include/pcl/gpu/kinfu/
)

# LIBRARIES
SET( MY_LIBRARIES
    ${PCL_LIBRARIES}
    #${PCL_LOC}/lib/libpcl_common.so
    #${PCL_LOC}/lib/libpcl_io.so
    #${PCL_LOC}/lib/libpcl_visualization.so
    #${PCL_LOC}/lib/libpcl_gpu_containers.so
    ${OpenCV_LIBRARIES}
    OpenNI
    boost_system
    boost_filesystem
    boost_thread
    vtkCommon
    vtkFiltering
    vtkRendering
    vtkImaging
    vtkGraphics
    vtkIO
    cudart
    ${CUDA_LIBRARIES}
    #${GLUT_LIBRARIES}
    #${OPENGL_LIBRARIES}
    #stdc++
    #pthread
)
