#include <opencv2/core/core.hpp>

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
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

#if 1
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
    return 0;
#elif 1

    MyImage<unsigned> myImage("myImage");
    cvImage2Array( img, myImage.Image(), myImage.width, myImage.height );
    printf("myImage.image: %X\n", myImage.Image() );

    MyImage<unsigned> myGuide("myGuide");
    cvImage2Array( guide, myGuide.Image(), myGuide.width, myGuide.height );

    mySingleRun( myImage, myGuide, argc, argv );
    return 0;
#endif

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
}
