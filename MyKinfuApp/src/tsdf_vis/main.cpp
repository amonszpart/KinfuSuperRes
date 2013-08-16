#include <iostream>
#include <pcl/console/parse.h>
#include <boost/filesystem.hpp>
#include <string>

#include "tsdf_viewer.h"
#include "my_pcd_viewer.h"
#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "BilateralFilterCuda.hpp"
#include "../../BilateralFilteringCuda/include/YangFiltering.h"

int testCostVolume( std::string depPath, std::string imgPath, int iterations = 3 )
{
    const int   L       = 20; // depth search range
    const float ETA     = .5f;
    const float ETA_L_2 = ETA*L*ETA*L;
    const float MAXRES  = 255.0f;

    // read
    //std::string path    = "/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130809_1415/";
    //std::string path    = "/home/bontius/workspace/cpp_projects/KinfuSuperRes/SuperRes-NI-1-5/build/out/imgs_20130809_1438/";
    //cv::Mat dep16       = cv::imread( path + "mapped16_00000001.png", -1 );
    //cv::Mat rgb8        = cv::imread( path + "img8_1280_00000001.png", -1 );
    cv::Mat dep16       = cv::imread( depPath, -1 );
    cv::Mat rgb8        = cv::imread( imgPath, -1 );

    cv::Mat dep8;         dep16.convertTo( dep8, CV_8UC1, 255.f / 10001.f );
    cv::Mat dep8_large;

    // show
    cv::imshow( "dep16", dep16 );
    cv::imshow( "dep8", dep8 );
    cv::imshow( "img8" , rgb8  );

    char key_pressed = 0;

    // crossFiltering
    static BilateralFilterCuda<float> cbfc;
    cbfc.setFillMode( FILL_ONLY_ZEROS );

    // select input depth
    float depMax = 255.f;
    cv::Mat &dep = dep8; // 640 by default
    // upscale
    if ( rgb8.size() != dep16.size() )
    {
        cv::resize( dep16, dep8_large, rgb8.size(), 0, 0, cv::INTER_NEAREST );
        dep = dep8_large;
        depMax = 10001.f;
    }

    // convert input depth to float
    cv::Mat fDep; dep.convertTo( fDep, CV_32FC1  );

    BilateralFilterCuda<float> bfc;
    bfc.setIterations( 5 );
    bfc.setFillMode( FILL_ONLY_ZEROS );
    cv::Mat bilfiltered;
    bfc.runBilateralFiltering( fDep, rgb8, bilfiltered,
                               5.f, .1f, 10, 1.f );
    cv::imshow( "bilf", bilfiltered / depMax );
    bilfiltered.copyTo( fDep );

#if 1
    YangFiltering::run( fDep, rgb8, fDep, iterations );
#elif 0
    // input: fDep(CV_32FC1,0..10001.f), rgb8(CV_8UC3)

    // prepare
    cv::Mat truncC2     ( fDep.size(), CV_32FC1 ); // C(d)
    cv::Mat truncC2_prev( fDep.size(), CV_32FC1 ); // C(d-1)
    cv::Mat minDs       ( fDep.size(), CV_32FC1 ); // d_min
    cv::Mat minC        ( fDep.size(), CV_32FC1 ); // C(d_min)
    cv::Mat minCm1      ( fDep.size(), CV_32FC1 ); // C(d_min-1)
    cv::Mat minCp1      ( fDep.size(), CV_32FC1 ); // C(d_min+1)

    for ( int it = 0; it < 5; ++it )
    {
        // select range of candidates
        double maxVal;
        cv::minMaxIdx( fDep, 0, &maxVal );
        std::cout << "max: " << maxVal << std::endl;

        minC  .setTo( maxVal * maxVal );
        minCm1.setTo( maxVal * maxVal );
        minCp1.setTo( maxVal * maxVal );

        for ( int d = 0; d < min((float)maxVal + L + 1, MAXRES); d+=1 )
        {
            std::cout << "d: " << d << " -> " << maxVal + L + 1 << std::endl;

            // calculate truncated cost
            MyThrustUtil::squareDiff( fDep, d, truncC2, ETA_L_2 );

            // filter cost slice
            cbfc.runBilateralFiltering( /*            in: */ truncC2,
                                        /*         guide: */ rgb8,
                                        /*           out: */ truncC2,
                                        /* spatial sigma: */ 1.5f,
                                        /*   range sigma: */ .03f,
                                        /*  kernel range: */ 12 );

            // track minimums
            MyThrustUtil::minMaskedCopy( truncC2_prev, truncC2, d, minC, minDs, minCm1, minCp1 );

            // show
            //cv::imshow( "minC" , minC / MAXRES / MAXRES );
            //cv::imshow( "minDs", minDs / depMax );

            truncC2.copyTo( truncC2_prev );

            //cv::waitKey(50);
        }

        // refine minDs based on C(d_min), C(d_min-1), C(d_min+1)
        MyThrustUtil::subpixelRefine( minC, minCm1, minCp1, minDs );

        // copy to output
        minDs.copyTo( fDep );
        cv::imshow( "fDep", fDep / MAXRES );
        cv::waitKey(50);
    }

    // output: fDep
#else
    // prepare output
    cv::Mat fDep_next( dep.rows, dep.cols, CV_32FC1 );
    // iterate
    for ( int it = 0; it < 5; ++it )
    {
        // select range of candidates
        double maxVal;
        cv::minMaxIdx( fDep, 0, &maxVal );
        std::cout << "max: " << maxVal << std::endl;

        // calculate cost volume for every depth candidate
        cv::Mat C      ( dep.size(), CV_32FC1 );                                // simple depth difference
        cv::Mat C2     ( dep.size(), CV_32FC1 );                                // squared depth difference
        cv::Mat truncC2( dep.size(), CV_32FC1 );                                // truncated squared depth difference
        cv::Mat minC   ( dep.size(), CV_32FC1 ); minC.setTo( maxVal * maxVal ); // minimum cost over d values
        cv::Mat minDs  ( dep.size(), CV_32FC1 );                                // d values for minimum costs
        for ( int d = 0; d < maxVal + L + 1; d+=1 )
        {
            // info
            std::cout << d << std::endl;

            // calculate cost slice
            cv::absdiff( fDep, d, C );
            cv::multiply( C, C, C2 );
            truncC2 = cv::min( C2, ETA_L_2 );

            // filter cost slice
            cbfc.runBilateralFiltering( /*            in: */ truncC2,
                                        /*         guide: */ rgb8,
                                        /*           out: */ truncC2,
                                        /* spatial sigma: */ 1.5f,
                                        /*   range sigma: */ .03f,
                                        /*  kernel range: */ 12 );
            //cv::imshow( "C2", C2 / 65536.f );
            //cv::imshow( "truncC2", truncC2 / ETA_L_2 / 2.f );

            /// track minimum cost d values
            {
                // replace minimum costs
                minC = cv::min( /*  stored: */ minC,
                                /* current: */ truncC2 );

                // selection of minimum places
                // minMask = (minC == truncC2)
                cv::Mat minMask;
                cv::compare( /*      new costs: */ minC,
                             /*  current costs: */ truncC2,
                             /* current places: */ minMask,
                             /* condition "==": */ CV_CMP_EQ );

                // minDepths( minC == truncC2 ) = current d;
                minDs.setTo( d, minMask );
            }

            key_pressed = cv::waitKey(7);
            if ( key_pressed == 27 )
                break;
        }
        cv::imshow( "minC" , minC / MAXRES / MAXRES );
        cv::imshow( "minDs", minDs );

        /// calculate costs of neighbour depths
        cv::Mat ftmp( dep.size(), CV_32FC1 );
        // d_-
        cv::Mat d_m1( dep.size(), CV_32FC1 );
        cv::subtract( minDs, 1.f, d_m1, cv::Mat(), CV_32FC1 );
        // d_+
        cv::Mat d_p1( dep.size(), CV_32FC1 );
        cv::add( minDs, 1.f, d_p1, cv::Mat(), CV_32FC1 );

        // f(d_-)
        cv::Mat f_d_m1;
        cv::absdiff( fDep, d_m1, ftmp );
        cv::multiply( ftmp, ftmp, f_d_m1 );
        f_d_m1 = cv::min( f_d_m1, ETA_L_2 );

        // f(d_+)
        cv::Mat f_d_p1( dep.rows, dep.cols, CV_32FC1 );
        cv::absdiff( fDep, d_p1, ftmp );
        cv::multiply( ftmp, ftmp, f_d_p1 );
        f_d_p1 = cv::min( f_d_p1, ETA_L_2 );

        /// subpixel
        cv::Mat a1 = ( f_d_p1 - f_d_m1 );
        cv::Mat a2 = ( 2.f * (f_d_p1 + f_d_m1 - 2.f * minC) );
        cv::Mat a3;
        cv::divide( a1, a2, a3, 1.0, CV_32FC1 );
        a3 = cv::min( a3, MAXRES/16.f );
        a3 = cv::max( -(MAXRES/16.f), a3 );
        cv::subtract( minDs, a3, fDep_next, cv::Mat(), CV_32FC1 );
#if 0
        //cv::imwrite( "a1", a1 );
        std::string dpath = "out/";
        util::writeCvMat2MFile<float>( a1       , dpath+"load_a1.m"       , "a1" );
        util::writeCvMat2MFile<float>( a2       , dpath+"load_a2.m"       , "a2" );
        util::writeCvMat2MFile<float>( a3       , dpath+"load_a3.m"       , "a3" );
        util::writeCvMat2MFile<float>( minDs    , dpath+"load_minDs.m"    , "minDs" );
        util::writeCvMat2MFile<float>( fDep_next, dpath+"load_fDep_next.m", "fDep_next" );
#endif


        {
            double minVal, maxVal;
            cv::minMaxIdx( fDep_next, &minVal, &maxVal );
            std::cout << "minVal(fDep_next): " << minVal << ", "
                      << "maxVal(fDep_next): " << maxVal << std::endl;
        }

        cv::imshow( "fDep_next", fDep_next / MAXRES );
        cv::waitKey(10);
        fDep_next.copyTo( fDep );
    }
#endif

    {
        double minVal, maxVal;
        cv::minMaxIdx( fDep, &minVal, &maxVal );
        std::cout << "minVal(fDep): " << minVal << ", "
                  << "maxVal(fDep): " << maxVal << std::endl;
        cv::Mat tmp;
        fDep.convertTo( tmp, CV_16UC1 );
        cv::imwrite( "yang16.png", tmp, (std::vector<int>){16,0} );
    }

    while ( key_pressed != 27 )
    {
        key_pressed = cv::waitKey();
    }

    return EXIT_SUCCESS;
}

struct MyPlayer
{
        MyPlayer()
            : exit(false), changed(false) {}

        bool exit;

        bool changed;

        Eigen::Affine3f& Pose() { changed = true; return pose; }
        Eigen::Affine3f const& Pose() const { return pose; }
    protected:
        Eigen::Affine3f pose;

} myPlayer;

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
keyboard_callback( const pcl::visualization::KeyboardEvent &e, void *cookie )
{
    MyPlayer* pMyPlayer = reinterpret_cast<MyPlayer*>( cookie );

    int key = e.getKeyCode ();

    if ( e.keyUp () )
    {
        switch ( key )
        {
            case 27:
                pMyPlayer->exit = true;
                break;
            case 82:
            case 'a':
                pMyPlayer->Pose().translation().x() -= 0.1f;
                break;
            case 'd':
            case 83:
                pMyPlayer->Pose().translation().x() += 0.1f;
                break;
            case 's':
            case 84:
                pMyPlayer->Pose().translation().y() -= 0.1f;
                break;
            case 'w':
            case 81:
                pMyPlayer->Pose().translation().y() += 0.1f;
                break;
            case 'e':
                pMyPlayer->Pose().translation().z() += 0.1f;
                break;
            case 'c':
                pMyPlayer->Pose().translation().z() -= 0.1f;
                break;

            default:
                break;
        }
    }
    std::cout << (int)key << std::endl;
}


void printUsage()
{
    std::cout << "Usage:\n\tTSDFVis --in cloud.dat\n" << std::endl;
}

// --in /home/amonszpart/rec/testing/keyboard_cross_nomap_20130811_1828/cloud_tsdf_volume.dat
int main( char argc, char** argv )
{
    std::string yangDir;
    bool doYang = pcl::console::parse_argument (argc, argv, "--yangd", yangDir) >= 0;
    if ( doYang )
    {
        std::string depName;
        doYang &= pcl::console::parse_argument (argc, argv, "--dep", depName) >= 0;
        std::string imgName;
        doYang &= pcl::console::parse_argument (argc, argv, "--img", imgName) >= 0;
        int iterations = 3;
        doYang &= pcl::console::parse_argument (argc, argv, "--iter", iterations);
        if ( doYang )
        {
            if ( iterations <= 0 )
                iterations = 3;
            std::cout << "running for " << iterations << std::endl;
            testCostVolume( yangDir + "/" + depName, yangDir + "/" + imgName, iterations );
        }
        else
        {
            std::cerr << "yang usage: --yangd dir --dep depName --img imgName" << std::endl;
            return EXIT_FAILURE;
        }

    }

    std::string tsdfFilePath;
    if (pcl::console::parse_argument (argc, argv, "--in", tsdfFilePath) < 0 )
    {
        printUsage();
        return 1;
    }

    //std::unique_ptr<am::MyPCDViewer> pcdViewer( new am::MyPCDViewer() );
    //pcdViewer->loadTsdf( pcdFilePath,  );
    //pcdViewer->run( pcdFilePath );


    // Load TSDF
    std::unique_ptr<am::TSDFViewer> tsdfViewer( new am::TSDFViewer() );
    tsdfViewer->loadTsdfFromFile( tsdfFilePath, true );
    tsdfViewer->getRayViewer()->registerKeyboardCallback (keyboard_callback, (void*)&myPlayer);
    tsdfViewer->getDepthViewer()->registerKeyboardCallback (keyboard_callback, (void*)&myPlayer);

    tsdfViewer->dumpMesh();
#if 0
    Eigen::Affine3f pose;
    pose.linear() <<  0.999154,  -0.0404336, -0.00822448,
            0.0344457,    0.927101,   -0.373241,
            0.0227151,    0.372638,    0.927706;
    pose.translation() << 1.63002,
            1.46289,
            0.227635;
#endif

    // 224
    //Eigen::Affine3f pose;
    myPlayer.Pose().translation() << 1.67395, 1.69805, 0.337846;

    myPlayer.Pose().linear() <<
                     0.954062,  0.102966, -0.281364,
                    -0.16198,  0.967283, -0.195268,
                    0.252052,  0.231873,  0.939525;

    while ( !myPlayer.exit )
    {
        if ( myPlayer.changed )
        {
            tsdfViewer->showGeneratedRayImage( tsdfViewer->kinfuVolume_ptr_, myPlayer.Pose() );
            tsdfViewer->showGeneratedDepth   ( tsdfViewer->kinfuVolume_ptr_, myPlayer.Pose() );
            myPlayer.changed = false;
        }
        tsdfViewer->spinOnce(30);
    }

    // get depth
    auto vshort = tsdfViewer->getLatestDepth();
    cv::Mat dep( 480, 640, CV_16UC1, vshort.data() );
    std::vector<int> params;
    params.push_back(16);
    params.push_back(0);
    cv::imwrite("dep224.png", dep, params );

    //tsdfViewer->spin();
    //tsdfViewer->toCloud( myPlayer.Pose() );

    std::cout << "Hello Tsdf_vis" << std::endl;


}
