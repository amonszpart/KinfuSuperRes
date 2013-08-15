project(AMOniRecorder)
cmake_minimum_required(VERSION 2.8)

SET( OR_NAME "AMOniRecorder" )
SET( CMAKE_CXX_COMPILER /usr/bin/gcc )
SET( CMAKE_CXX_FLAGS "-std=c++0x" )
add_definitions( -D__x86_64__=1 -Dlinux=1 -DHAVE_OPENCV=1 -DHAVE_OPENNI=1 )

SET(TROLL TRUE)

if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
    SET(OPENNI_INCLUDE /usr/include/openni)

    SET( CUDA_LIB cuda )
    SET( GLEW_LIB GLEW )

    SET( CMAKE_CXX_COMPILER "/usr/bin/gcc-4.4" )
    SET( CUDA_HOST_COMPILER "/usr/bin/gcc-4.4" )
else(TROLL) # Ubul
    #SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )

    SET( CUDA_LIB /usr/lib/nvidia-310-updates/libcuda.so )
    SET( GLEW_LIB /usr/lib/x86_64-linux-gnu/libGLEW.so )
    #SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
    #SET(OPENNI_INCLUDE /usr/include/ni)

    SET( CMAKE_CXX_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/gcc" )
    SET( CUDA_HOST_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/gcc" )
endif(TROLL)

FIND_PACKAGE( OpenCV REQUIRED )


SET( OR_HEADER_LIST
    Recorder.h
    ../util/MaUtil.h
    CvImageDumper
)

SET( OR_SRC_LIST
    ../util/MaUtil.cpp
    CvImageDumper
    Recorder.cpp
    main.cpp
)

# INCLUDES
INCLUDE_DIRECTORIES( ${OPENNI_INCLUDE} )

# EXECUTABLE
ADD_EXECUTABLE( ${OR_NAME} ${OR_SRC_LIST} ${OR_HEADER_LIST} )

# LINK
TARGET_LINK_LIBRARIES( ${OR_NAME}
    -lOpenNI
    ${OpenCV_LIBS}
)
