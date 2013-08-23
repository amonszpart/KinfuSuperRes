#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <helper_functions.h>  // CUDA SDK Helper functions

#include "ViewPointMapperCuda.h"
#include "BilateralFilterCuda.hpp"
#include "MyThrustUtil.h"

/*
 * @brief       Testruns ViewPointMapperCuda::runViewPointMapping
 * @param img16 16bit depth image to map
 * @param guide 8UC3 rgb image to map over
 **/
int testViewpointMapping( cv::Mat const& dep16, cv::Mat const& guide )
{
    cv::imshow( "img", dep16 );
    cv::Mat mapped16( cv::Mat::eye(dep16.size(), CV_32FC1 ) );
    ViewPointMapperCuda::runViewpointMapping( dep16, mapped16 );
    cv::imshow( "out", mapped16 );

    cv::Mat mappedF;
    mapped16.convertTo( mappedF, CV_32FC1, 1.f / 10001.f );
    {
        double minVal, maxVal;
        cv::minMaxIdx( mappedF, &minVal, &maxVal );
        std::cout << "minVal(fmapped16): " << minVal << ", "
                  << "maxVal(fmapped16): " << maxVal << std::endl;
    }
    cv::Mat mapped8;
    mapped16.convertTo( mapped8, CV_8UC1, 255.f / 10001.f );
    cv::imshow( "mapped8", mapped8 );

    cv::Mat depF;
    dep16.convertTo( depF, CV_32FC1, 1.f / 10001.f );
    {
        double minVal, maxVal;
        cv::minMaxIdx( depF, &minVal, &maxVal );
        std::cout << "minVal(depF): " << minVal << ", "
                  << "maxVal(depF): " << maxVal << std::endl;
    }


    cv::Mat diff;
    cv::absdiff( depF, mappedF, diff );
    cv::imshow( "diff", diff );

    cv::Mat fGuide;
    guide.convertTo( fGuide, CV_32FC1, 1.f / 255.f );

    std::vector<cv::Mat> fMappedVector;
    fMappedVector.push_back( mappedF );
    fMappedVector.push_back( mappedF );
    fMappedVector.push_back( mappedF );

    cv::Mat fMapped3;
    //cv::merge( fMappedVector, fMapped3 );
    cv::merge( fMappedVector.data(), 3, fMapped3 );

    cv::Mat fBlended( mappedF.rows, mappedF.cols, CV_32FC1 );
    cv::addWeighted( fGuide, 0.5,
                     fMapped3, 0.7, 0.0, fBlended, CV_32FC1 );
    cv::imshow( "fBlended", fBlended );
    cv::imwrite( "blended.bmp", fBlended * 255.0 );

    // test ushort version
    {
        ushort* data = new ushort[ dep16.cols * dep16.rows ];
        for ( int y = 0; y < dep16.rows; ++y )
        {
            for ( int x = 0; x < dep16.cols; ++x )
            {
                data[ y * dep16.cols + x ] = dep16.at<ushort>( y, x );
            }
        }

        ViewPointMapperCuda::runViewpointMapping( data, dep16.cols, dep16.rows );

        cv::Mat outMat( dep16.size(), dep16.type() );
        for ( int y = 0; y < dep16.rows; ++y )
        {
            for ( int x = 0; x < dep16.cols; ++x )
            {
                outMat.at<ushort>( y, x ) = data[ y * outMat.cols + x ];
            }
        }

        cv::imshow( "ushortMapping", outMat );

        SAFE_DELETE_ARRAY( data );
    }


    char c = 0;
    while ( (c = cv::waitKey()) != 27 ) ;


    return EXIT_SUCCESS;
}

/*
 * @brief       Testruns ViewPointMapperCuda::runViewPointMapping
 * @param img16 16bit depth image to map
 * @param guide 8UC3 rgb image to map over
 **/
int testBilateralFiltering( cv::Mat const& img16, cv::Mat const& guide )
{
    cv::imshow( "img16", img16 );

    /*cv::Mat bFiltered16;
    BilateralFilterCuda<float> bilateralFilterCuda;
    bilateralFilterCuda.runBilateralFiltering( img16, cv::Mat(), bFiltered16 );
    cv::imshow( "bfiltered16", bFiltered16 );*/

    cv::Mat cFiltered16;
    BilateralFilterCuda<float> crossBilateralFilterCuda;
    crossBilateralFilterCuda.runBilateralFiltering( img16, guide, cFiltered16 );
    cv::imshow( "cbfiltered16", cFiltered16 );

    cv::Mat filtered8;
    cFiltered16.convertTo( filtered8, CV_8UC1, 255.f / 10001.f );
    cv::imshow( "filtered8", filtered8 );

    cv::Mat fFiltered;
    cFiltered16.convertTo( fFiltered, CV_32FC1, 1.f / 10001.f );

    cv::Mat fImg;
    img16.convertTo( fImg, CV_32FC1, 1.f / 10001.f );

    cv::Mat diff;
    cv::absdiff( fImg, fFiltered, diff );
    cv::imshow( "diff", diff );

    cv::Mat fGuide;
    guide.convertTo( fGuide, CV_32FC1, 1.f / 255.f );

    std::vector<cv::Mat> fMappedVector;
    fMappedVector.push_back( fFiltered );
    fMappedVector.push_back( fFiltered );
    fMappedVector.push_back( fFiltered );

    cv::Mat fMapped3;
    cv::merge( fMappedVector.data(), 3, fMapped3 );
    //cv::merge( fMappedVector, fMapped3 );

    cv::Mat fBlended( fFiltered.rows, fFiltered.cols, CV_32FC1 );
    cv::addWeighted( fGuide, 0.5,
                     fMapped3, 0.7, 0.0, fBlended, CV_32FC1 );
    cv::imshow( "fBlended", fBlended );
    cv::imwrite( "blended.bmp", fBlended * 255.0 );

    cv::waitKey();
    return EXIT_SUCCESS;
}

#if CV_MINOR_VERSION < 4
namespace cv
{
    enum
    {
        // 8bit, color or not
        IMREAD_UNCHANGED  =-1,
        // 8bit, gray
        IMREAD_GRAYSCALE  =0,
        // ?, color
        IMREAD_COLOR      =1,
        // any depth, ?
        IMREAD_ANYDEPTH   =2,
        // ?, any color
        IMREAD_ANYCOLOR   =4
    };

    enum
    {
        IMWRITE_JPEG_QUALITY =1,
        IMWRITE_PNG_COMPRESSION =16,
        IMWRITE_PNG_STRATEGY =17,
        IMWRITE_PNG_STRATEGY_DEFAULT =0,
        IMWRITE_PNG_STRATEGY_FILTERED =1,
        IMWRITE_PNG_STRATEGY_HUFFMAN_ONLY =2,
        IMWRITE_PNG_STRATEGY_RLE =3,
        IMWRITE_PNG_STRATEGY_FIXED =4,
        IMWRITE_PXM_BINARY =32
    };
}
#endif

float testEqual( cv::Mat const& C1, cv::Mat const& C2 )
{
    int ok = 0;
    for ( int y = 0; y < C1.rows; ++y )
        for ( int x = 0; x < C1.cols; ++x )
        {
            if ( C1.at<float>(y,x) != C2.at<float>(y,x) )
                std::cout << C1.at<float>(y,x) << " "
                          << C2.at<float>(y,x) << " "
                          << C1.at<float>(y,x) - C2.at<float>(y,x)
                          << std::endl;
            else
                ++ok;
        }
    return float(ok) / (float)C1.cols / (float)C1.rows * 100.f;
}

#if 0
int testThrust( cv::Mat const& img16, cv::Mat const& guide )
{
    cv::Mat fImg16;
    img16.convertTo( fImg16, CV_32FC1 );
    cv::Mat fGuide;
    img16.convertTo( fImg16, CV_32FC1 );

    float d = 5.f;

    /// squareDiff
    // gpu
    cv::Mat C2_gpu;
    MyThrustUtil::squareDiff( fImg16, d, C2_gpu );
    // cpu
    cv::Mat C, C2_cpu;
    cv::absdiff( fImg16, d, C );
    cv::multiply( C, C, C2_cpu );
    // test
    std::cout << "squareDiffScalar: " << testEqual( C2_gpu, C2_cpu ) << "%" << std::endl;

    /// squareDiffTrunc
    double minVal, maxVal;
    cv::minMaxIdx( fImg16, &minVal, &maxVal );
    double truncAt = maxVal/2.f;
    // gpu
    cv::Mat truncC_gpu;
    MyThrustUtil::squareDiff( fImg16, d, truncC_gpu, truncAt );
    // cpu
    cv::Mat truncC_cpu = cv::min( C2_cpu, truncAt );
    // test
    std::cout << "squareDiffScalarTrunc: " << testEqual( truncC_gpu, truncC_cpu ) << "%" << std::endl;

    /// minMaskedCopy
    // prep
    cv::Mat minC ( fImg16.size(), CV_32FC1 ); minC.setTo( maxVal * maxVal );
    cv::Mat minCm1_gpu ( fImg16.size(), CV_32FC1 ); minCm1_gpu.setTo( maxVal * maxVal );
    cv::Mat minCp1_gpu ( fImg16.size(), CV_32FC1 ); minCp1_gpu.setTo( maxVal * maxVal );
    cv::Mat minDs_gpu( fImg16.size(), CV_32FC1 );
    cv::Mat minDs_cpu( fImg16.size(), CV_32FC1 );
    // gpu
    MyThrustUtil::minMaskedCopy( truncC_gpu, truncC_gpu, d, minC, minDs_gpu, minCm1_gpu, minCp1_gpu );
    // cpu
    minC = cv::min( minC, truncC_cpu );
    cv::Mat minMask;
    cv::compare( minC, truncC_cpu, minMask, CV_CMP_EQ );
    minDs_cpu.setTo( d, minMask );
    // test (CV_CMP_EQ makes this "<=", gpu implementation is "<", that's more correct)
    std::cout << "minMaskedCopy: " << testEqual( minDs_gpu, minDs_cpu ) << "%" << std::endl;

    /// estimateSubpixel
    // gpu
    MyThrustUtil::subpixelRefine( minC, minCm1_gpu, minCp1_gpu, minDs_gpu );
    std::cout << "suppixelRefine finished..." << std::endl;
    return 0;
    // cpu
    {
        cv::Mat fImg16;
        cv::Mat ftmp( fImg16.size(), CV_32FC1 );
        // d_-
        cv::Mat d_m1( fImg16.size(), CV_32FC1 );
        cv::subtract( minDs_cpu, 1.f, d_m1, cv::Mat(), CV_32FC1 );
        // d_+
        cv::Mat d_p1( fImg16.size(), CV_32FC1 );
        cv::add( minDs_cpu, 1.f, d_p1, cv::Mat(), CV_32FC1 );

        // f(d_-)
        cv::Mat f_d_m1;
        cv::absdiff( fImg16, d_m1, ftmp );
        cv::multiply( ftmp, ftmp, f_d_m1 );
        f_d_m1 = cv::min( f_d_m1, truncAt );

        // f(d_+)
        cv::Mat f_d_p1( fImg16.rows, fImg16.cols, CV_32FC1 );
        cv::absdiff( fImg16, d_p1, ftmp );
        cv::multiply( ftmp, ftmp, f_d_p1 );
        f_d_p1 = cv::min( f_d_p1, truncAt );

        /// subpixel
        cv::Mat a1 = ( f_d_p1 - f_d_m1 );
        cv::Mat a2 = ( 2.f * (f_d_p1 + f_d_m1 - 2.f * minC) );
        cv::Mat a3;
        cv::divide( a1, a2, a3, 1.0, CV_32FC1 );
        cv::subtract( minDs_cpu, a3, minDs_cpu, cv::Mat(), CV_32FC1 );
    }
    std::cout << "subpixelRefine: " << testEqual( minDs_gpu, minDs_cpu ) << "%" << std::endl;
}
#endif


////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    // --in /home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130805_1644/dep16_00000000.pgm --guide /home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130805_1644/img8_00000000.png

    char *image_path = NULL;
    char *guide_path = NULL;
    cv::Mat img16, guide;

    // image name
    if (checkCmdLineFlag(argc, (const char **)argv, "in"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "in", &image_path );
        img16 = cv::imread( image_path, cv::IMREAD_UNCHANGED );

    }
    else
    {
        std::cerr << "need to provide input file by '--in filename' argument" << std::endl;
        exit( EXIT_FAILURE );
    }

    // guide name
    if (checkCmdLineFlag(argc, (const char **)argv, "guide"))
    {
        getCmdLineArgumentString(argc, (const char **)argv, "guide", &guide_path );
        guide = cv::imread( guide_path, cv::IMREAD_UNCHANGED );

    }
    else
    {
        std::cerr << "need to provide guide file by '--guide filename' argument" << std::endl;
        exit( EXIT_FAILURE );
    }

    //return testThrust(img16,guide);

#if 1 // VIEWPOINTMAPPING
    return testViewpointMapping( img16, guide );
#elif 1 // BilateralFiltering
    return testBilateralFiltering( img16, guide );
#elif 1 // BilateralFiltering old

    MyImage<unsigned> myImage("myImage");
    cvImage2Array( img, myImage.Image(), myImage.width, myImage.height );
    printf("myImage.image: %X\n", myImage.Image() );

    MyImage<unsigned> myGuide("myGuide");
    cvImage2Array( guide, myGuide.Image(), myGuide.width, myGuide.height );

    mySingleRun( myImage, myGuide, argc, argv );
    return 0;
#endif

#if 0
    // start logs
    int devID;
    char *ref_file = NULL;
    printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    if (argc > 1)
    {
        if (checkCmdLineFlag(argc, (const char **)argv, "radius"))
        {
            filter_radius = getCmdLineArgumentInt(argc, (const char **) argv, "radius");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "passes"))
        {
            iterations = getCmdLineArgumentInt(argc, (const char **)argv, "passes");
        }

        if (checkCmdLineFlag(argc, (const char **)argv, "file"))
        {
            getCmdLineArgumentString(argc, (const char **)argv, "file", (char **)&ref_file);
        }
    }

    // load image to process
    loadImageData(argc, argv);

    if (checkCmdLineFlag(argc, (const char **)argv, "benchmark"))
    {
        // This is a separate mode of the sample, where we are benchmark the kernels for performance
        devID = findCudaDevice(argc, (const char **)argv);

        // Running CUDA kernels (bilateralfilter) in Benchmarking mode
        g_TotalErrors += runBenchmark(argc, argv);
        exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    else if (checkCmdLineFlag(argc, (const char **)argv, "radius") ||
             checkCmdLineFlag(argc, (const char **)argv, "passes"))
    {
        // This overrides the default mode.  Users can specify the radius used by the filter kernel
        devID = findCudaDevice(argc, (const char **)argv);
        g_TotalErrors += runSingleTest(ref_file, argv[0]);
        exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
    else
    {
        // Default mode running with OpenGL visualization and in automatic mode
        // the output automatically changes animation
        printf("\n");

        // First initialize OpenGL context, so we can properly set the GL for CUDA.
        // This is necessary in order to achieve optimal performance with OpenGL/CUDA interop.
        initGL(argc, (char **)argv);
        int dev = findCapableDevice(argc, argv);

        if (dev != -1)
        {
            dev = gpuGLDeviceInit(argc, (const char**)argv);
            if( dev == -1 ) {
                exit(EXIT_FAILURE);
            }
        }
        else
        {
            cudaDeviceReset();
            exit(EXIT_SUCCESS);
        }

        // Now we can create a CUDA context and bind it to the OpenGL context
        initCuda<unsigned>( hImage, width, height );
        initGLResources();

        // sets the callback function so it will call cleanup upon exit
        atexit(cleanup);

        printf("Running Standard Demonstration with GLUT loop...\n\n");
        printf("Press '+' and '-' to change filter width\n"
               "Press ']' and '[' to change number of iterations\n"
               "Press 'e' and 'E' to change Euclidean delta\n"
               "Press 'g' and 'G' to changle Gaussian delta\n"
               "Press 'a' or  'A' to change Animation mode ON/OFF\n\n");

        // Main OpenGL loop that will run visualization for every vsync
        glutMainLoop();

        cudaDeviceReset();
        exit(g_TotalErrors == 0 ? EXIT_SUCCESS : EXIT_FAILURE);
    }
#endif
}
