#include "YangFilteringWrapper.h"

#include "YangFiltering.h"
//#include "../../BilateralFilteringCuda/include/YangFiltering.h"

#include "BilateralFilterCuda.hpp"
#include "ViewPointMapperCuda.h"

#include <boost/filesystem.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <boost/filesystem.hpp>
#include "AMUtil2.h"
#include "MaUtil.h"

namespace am {

    int runYangCleaned( /* out: */ cv::Mat &filteredDep16,
                        /*  in: */ std::string depPath, std::string imgPath, YangFilteringRunParams yangFilteringRunParams, std::string const& path )
    {
        cv::Mat dep16;
        // read
        if ( depPath.find("png") == std::string::npos )
        {
            if ( depPath.find("pfm") == std::string::npos )
            {
                std::cerr << "not png, not pfm, what is it??" << std::endl;
                return 1;
            }

            am::util::loadPFM( dep16, depPath );
        }
        else
        {
            dep16 = cv::imread( depPath, -1 );
        }

        MyIntrinsicsFactory factory;

        cv::Mat undistorted_dep;
        ViewPointMapperCuda::runViewpointMapping(
                    /*   in: */ dep16,
                /*      out: */ undistorted_dep,
                /* dep_intr: */ factory.createIntrinsics( DEP_CAMERA, false ), // depth is already undistorted in kinfu
                /* rgb_intr: */ factory.createIntrinsics( rgb_intr.at<float>(0,0), // don't distort rgb, it will be undistorted later
                                                          rgb_intr.at<float>(1,1),
                                                          rgb_intr.at<float>(0,2),
                                                          rgb_intr.at<float>(1,2) )
                );
        factory.clear();

        // undistort rgb
        cv::Mat rgb8        = cv::imread( imgPath, -1 );
        {
            std::cout << "YangFilteringWrapper: undistorting rgb with size: "
                      << rgb8.cols << "x" << rgb8.rows
                      << " with intrinsics: 1280x1024"
                      << std::endl;
            cv::Mat tmp;
            ViewPointMapperCuda::undistortRgb( /* out: */ tmp,
                                               /*  in: */ rgb8,
                                               am::viewpoint_mapping::INTR_RGB_1280_1024,
                                               am::viewpoint_mapping::INTR_RGB_1280_1024 );
            //cv::resize( tmp, rgb8, cv::Size(1280,960),0,0, cv::INTER_LANCZOS4 );
            tmp.copyTo( rgb8 );
        }


        if ( dep16.empty() || rgb8.empty() )
        {
            std::cerr << "YangFilteringWrapper::runYang(): dep16 or rgb8 empty...exiting..." << std::endl;
            return EXIT_FAILURE;
        }

        return runYangCleaned( filteredDep16, dep16, rgb8, yangFilteringRunParams, path );
    }

    int runYangCleaned( cv::Mat &filteredDep16, cv::Mat const& dep16, cv::Mat const& rgb8, YangFilteringRunParams yangFilteringRunParams, std::string const& path )
    {
        int res = EXIT_SUCCESS;

        // resize
        cv::Mat dep16_large;
        if ( rgb8.size() != dep16.size() )
        {
            std::cout << "YangFilteringWrapper::runYangCleaned(): upsizing depth to match rgb size... " << rgb8.cols << "x" << rgb8.rows << std::endl;

            cv::resize( dep16, dep16_large, rgb8.size(), 0, 0, cv::INTER_NEAREST );
        }
        cv::Mat const& depRef = dep16_large.empty() ? dep16 : dep16_large; // 640 by default

        // to float
        cv::Mat depFC1; depRef.convertTo( depFC1, CV_32FC1  );

        // bilateral fill
        BilateralFilterCuda<float> bfc;
        bfc.setIterations( 2 );
        bfc.setFillMode( FILL_ONLY_ZEROS );
        bfc.runBilateralFiltering( depFC1, rgb8, depFC1,
                                   5.f, .1f, 10, 1.f );
        // Yang
        YangFiltering yf;
        res += yf.run( depFC1, rgb8, depFC1, yangFilteringRunParams, path );

        //if ( dep16.type() == CV_16UC1 ) depFC1.convertTo( filteredDep16, CV_16UC1 );
        //else
        depFC1.copyTo( filteredDep16 );

        return res;
    }

    // cd ~/rec/testing/ram*/poses/bruteYang &&
    // mkdir ../safe &&
    // ~/cpp_projects/KinfuSuperRes/Tsdf_vis/bruteYang.sh ".." "d" "png"
    int bruteRun( std::string depPath, std::string imgPath )
    {
        std::cout << "C++BruteYang depPath: " << depPath << " imgPath: " << imgPath << std::endl;

        YangFilteringRunParams params;

        params.yang_iterations = 50;
        for ( float spatial_sigma = 1.0f; spatial_sigma < 1.9f; spatial_sigma += .1f )
        {
            params.spatial_sigma = spatial_sigma;
            for ( float range_sigma = .01f; range_sigma < .15f; range_sigma += .02f )
            {
                params.range_sigma = range_sigma;
                for ( int kernel_range = 1; kernel_range < 5; ++kernel_range )
                {
                    params.kernel_range = kernel_range;
                    std::cout << "params.kernel_range: " << params.kernel_range << std::endl;
                    std::cout << "params.range_sigma: " << params.range_sigma << std::endl;
                    std::cout << "params.spatial_sigma: " << params.spatial_sigma << std::endl;

                    runYang( depPath, imgPath, params );

                    char command[1024];
                    sprintf( command, "tar -zcvf ../safe/bruteYang_img_%s_ss_%2.3f_rs_%2.3f_kr_%d.tar.gz ./*.png --remove-files",
                             boost::filesystem::path(depPath).stem().string().c_str(),
                             params.spatial_sigma,
                             params.range_sigma,
                             params.kernel_range );
                    std::cout << "running..." << command << (system(command) ? "\tOK" : "\tfailed...") << std::endl;
                }
            }
        }
        //spatial_sigma( 1.1f ),
        //range_sigma( .03f ),
        //kernel_range( 4 ),
    }

    int runYang( std::string depPath, std::string imgPath, YangFilteringRunParams yangFilteringRunParams )
    {
        const int   L       = 20; // depth search range
        const float ETA     = .5f;
        const float ETA_L_2 = ETA*L*ETA*L;
        const float MAXRES  = 255.0f;

        // read
        cv::Mat dep16       = cv::imread( depPath, -1 );
        cv::Mat rgb8        = cv::imread( imgPath, -1 );
        if ( dep16.empty() || rgb8.empty() )
        {
            std::cerr << "YangFilteringWrapper::runYang(): dep16 or rgb8 empty...exiting..." << std::endl;
            return EXIT_FAILURE;
        }

        cv::Mat dep8;         dep16.convertTo( dep8, CV_8UC1, 255.f / 10001.f );
        cv::Mat dep8_large;

        // show
        //cv::imshow( "dep16", dep16 );
        //cv::imshow( "dep8", dep8 );
        //cv::imshow( "img8" , rgb8  );

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

        /*BilateralFilterCuda<float> bfc;
        bfc.setIterations( 2 );
        bfc.setFillMode( FILL_ONLY_ZEROS );
        cv::Mat bilfiltered;
        bfc.runBilateralFiltering( fDep, rgb8, bilfiltered,
                                   5.f, .1f, 10, 1.f );
        //cv::imshow( "bilf", bilfiltered / depMax );
        bilfiltered.copyTo( fDep );*/

    boost::filesystem::path path = boost::filesystem::path(depPath);
    std::cout << "path: " << path.parent_path().string();

#if 1
    YangFiltering yf;
    yf.run( fDep, rgb8, fDep, yangFilteringRunParams, path.parent_path().string() + "/" );
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
            cv::imwrite( depPath + "yang16.png", tmp, (std::vector<int>){16,0} );
            fDep.convertTo( tmp, CV_8UC1 );
            cv::imwrite( depPath + "/yang8.png", tmp, (std::vector<int>){16,0} );
        }

        /*while ( key_pressed != 27 )
        {
            key_pressed = cv::waitKey();
        }*/

        return EXIT_SUCCESS;
    }


} // end ns am
