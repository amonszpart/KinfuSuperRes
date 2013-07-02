    SET( PCL_LOC ~/3rdparty/pcl-trunk/build/install )
    SET( CUDA_PATH /usr/local/cuda-5.0 )
    SET( OPENNI_LOC ~/3rdparty/OpenNI-1.5/ )

HEADERS += \
    src/MaCvImageBroadcaster.h \
    src/HomographyCalculator.h \
    src/BilateralFiltering.h \
    src/amCommon.h \
    src/io/Recorder.h \
    src/io/CvImageDumper.h \
    src/util/XnVUtil.h \
    src/util/MaUtil.h

SOURCES += \
    src/main.cpp \
    src/MaCvImageBroadcaster.cpp \
    src/HomographyCalculator.cpp \
    src/BilateralFiltering.cpp \
    src/io/Recorder.cpp \
    src/io/CvImageDumper.cpp \
    src/util/MaUtil.cpp

OTHER_FILES += \
    CMakeLists.txt
