project(BilateralFilteringCuda)
cmake_minimum_required(VERSION 2.8)

#INCLUDE(localcfg.cmake)
#add_subdirectory( src/kinfu_large_scale )

# FLAGS
SET( CMAKE_CXX_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/gcc" )
SET( CUDA_HOST_COMPILER "/usr/lib/nvidia-cuda-toolkit/bin/gcc" )
SET( CUDA_NVCC_FLAGS "-arch=sm_11" )

#set(CUDA_SEPARABLE_COMPILATION ON)
#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
#set(BUILD_SHARED_LIBS OFF)
#set(CUDA_SEPARABLE_COMPILATION ON)
#SET( CMAKE_CXX_FLAGS "-std=c++0x" )

# PARAMS
if(TROLL)
    SET( PCL_LOC ~/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )
    SET(OPENNI_INCLUDE /usr/include/openni)
else(TROLL) # Ubul
    #SET( PCL_LOC ~/workspace/3rdparty/pcl-trunk/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    #SET( OPENNI_LOC ~/workspace/3rdparty/OpenNI-Bin-Dev-Linux-x64-v1.5.4.0/)
    #SET(OPENNI_INCLUDE /usr/include/ni)
endif(TROLL)

# PACKAGES
FIND_PACKAGE( OpenCV REQUIRED )
find_package( CUDA REQUIRED )

#FIND_PACKAGE( PCL 1.3 REQUIRED COMPONENTS common io )
#link_directories(${PCL_LIBRARY_DIRS})

# DEFS
add_definitions( -D__x86_64__=1 -Dlinux=1)
#add_definitions( ${PCL_DEFINITIONS} )

# BUILD
set( dir ${CMAKE_CURRENT_SOURCE_DIR}/build)
set( EXECUTABLE_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)
set( LIBRARY_OUTPUT_PATH ${dir} CACHE PATH "Build directory" FORCE)

MESSAGE( STATUS "CUDA_INCLUDE_DIRECTORIES: " ${CUDA_INCLUDE_DIRECTORIES})
MESSAGE( STATUS "CUDA_LIBRARY_DIRS: " ${CUDA_LIBRARY_DIRS} )
MESSAGE( STATUS "CUDA_INC_PATH: " ${CUDA_INC_PATH} )

# INCLUDES
SET( MY_INCLUDES
    inc
    #/usr/include/eigen3/
    #/usr/include/vtk-5.8/
    #${CUDA_PATH}/include
    #${OPENNI_INCLUDE}
)

# LIBRARIES
SET( MY_LIBRARIES
    m
    stdc++
    glut
    GL
    /usr/lib/nvidia-310-updates/libcuda.so
    /usr/lib/x86_64-linux-gnu/libGLEW.so
    #/usr/lib/x86_64-linux-gnu/libcudart.so.5.0
    #/usr/local/cuda-5.0/lib/libcublas.so
    ${OpenCV_LIBS}
    #OpenNI
    #boost_system
    #boost_filesystem
    #boost_thread
    #vtkCommon
    #vtkFiltering
    #vtkRendering
)

IF(USE_PCL)
    SET( MY_INCLUDES
        ${MY_INCLUDES}
        ${PCL_LOC}/include/pcl-1.7
        ${PCL_LOC}/include/pcl-1.7/pcl/visualization
    )

    SET( MY_LIBRARIES
        ${MY_LIBRARIES}
        ${PCL_LOC}/lib/libpcl_common.so
        ${PCL_LOC}/lib/libpcl_io.so
        ${PCL_LOC}/lib/libpcl_visualization.so
        ${PCL_LOC}/lib/libpcl_gpu_kinfu.so
        ${PCL_LOC}/lib/libpcl_gpu_containers.so
    )

endif(USE_PCL)

######################## PROJECT files ##############################

SET( HEADERS_LIST
    src/ViewPointMapperCuda.h
    src/BilateralFilterCuda.h
    src/GpuImage.h
    src/GpuDepthMap.h
    src/GpuExecutionStream.h
    src/cutil_math.cuh
    src/AmCudaHelper.cuh
    src/AmCudaUtil.h
)

set( CUFILES
    src/ViewPointMapperCuda.cu
    src/BilateralFilterCuda.cu
    src/GpuImage.cu
    src/GpuDepthMap.cu
    src/AMCudaUtil.cu
)

# SRC
SET( CCFILES
    src/ViewPointMapperCuda.cpp
    src/BilateralFilterCuda.cpp
    src/AmCudaUtil.cpp
    #src/bilateralFilter_cpu.cpp
    #src/bmploader.cpp
    src/main.cpp
)

# INCLUDES
INCLUDE_DIRECTORIES( ${MY_INCLUDES} )

# EXECUTABLE
#ADD_EXECUTABLE( ${PROJECT_NAME} ${SRC_LIST} ${HEADERS_LIST} )
source_group("CUDA Source Files" FILES ${CUFILES})

#cuda_add_executable( cu${PROJECT_NAME} ${CCFILES} ${CUFILES})
cuda_add_library( cu${PROJECT_NAME} ${CCFILES} ${CUFILES})

# LINK
TARGET_LINK_LIBRARIES( cu${PROJECT_NAME} ${MY_LIBRARIES} )

