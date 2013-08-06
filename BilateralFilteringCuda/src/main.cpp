#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <helper_functions.h>  // CUDA SDK Helper functions

#include "ViewPointMapperCuda.h"
#include "BilateralFilterCuda.h"

/*
 * @brief       Testruns ViewPointMapperCuda::runViewPointMapping
 * @param img16 16bit depth image to map
 * @param guide 8UC3 rgb image to map over
 **/
int testViewpointMapping( cv::Mat const& img16, cv::Mat const& guide )
{

    cv::imshow( "img", img16 );
    cv::Mat mapped16( cv::Mat::eye(img16.size(), CV_32FC1 ) );
    ViewPointMapperCuda::runViewpointMapping( img16, mapped16 );
    cv::imshow( "out", mapped16 );

    cv::Mat fMapped;
    mapped16.convertTo( fMapped, CV_32FC1, 1.f / 10001.f );

    cv::Mat fImg;
    img16.convertTo( fImg, CV_32FC1, 1.f / 10001.f );

    cv::Mat diff;
    cv::absdiff( fImg, fMapped, diff );
    cv::imshow( "diff", diff );

    cv::Mat fGuide;
    guide.convertTo( fGuide, CV_32FC1, 1.f / 255.f );

    std::vector<cv::Mat> fMappedVector;
    fMappedVector.push_back( fMapped );
    fMappedVector.push_back( fMapped );
    fMappedVector.push_back( fMapped );

    cv::Mat fMapped3;
    cv::merge( fMappedVector, fMapped3 );

    cv::Mat fBlended( fMapped.rows, fMapped.cols, CV_32FC1 );
    cv::addWeighted( fGuide, 0.5,
                     fMapped3, 0.7, 0.0, fBlended, CV_32FC1 );
    cv::imshow( "fBlended", fBlended );
    cv::imwrite( "blended.bmp", fBlended * 255.0 );

    cv::waitKey();
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

    cv::Mat bFiltered16;
    BilateralFilterCuda bilateralFilterCuda;
    bilateralFilterCuda.runBilateralFiltering( img16, cv::Mat(), bFiltered16 );
    cv::imshow( "bfiltered16", bFiltered16 );

    cv::Mat cFiltered16;
    BilateralFilterCuda crossBilateralFilterCuda;
    crossBilateralFilterCuda.runBilateralFiltering( bFiltered16, guide, cFiltered16 );
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
    cv::merge( fMappedVector, fMapped3 );

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

#if 0 // VIEWPOINTMAPPING
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
