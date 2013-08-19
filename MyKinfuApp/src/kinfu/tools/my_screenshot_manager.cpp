/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2010-2011, Willow Garage, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  Author: Francisco Heredia, Technical University Eindhoven, (f.j.mysurname.soriano < aT > tue.nl)
 */

#ifndef MY_SCREENSHOT_MANAGER_CPP_
#define MY_SCREENSHOT_MANAGER_CPP_

#include "my_screenshot_manager.h"
#include <pcl/io/png_io.h>
#include <fstream>
#include <map>

namespace am
{

    MyScreenshotManager::MyScreenshotManager()
    {
        path_ = "KinFuSnapshots";
        screenshot_counter = 0;
        setCameraIntrinsics();
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void
    MyScreenshotManager::saveImage( const Eigen::Affine3f &camPose, const pcl::gpu::PtrStepSz<const PixelRGB> &rgb24, const pcl::gpu::PtrStepSz<const unsigned short> &depth16 )
    {

        if ( !boost::filesystem::exists(path_) )
             boost::filesystem::create_directory( path_ );

        PCL_WARN ("[o] [o] [o] [o] Saving screenshot [o] [o] [o] [o]\n");

        std::string filename_image = path_ + "/";
        std::string filename_depth = path_ + "/d";
        std::string file_extension_image = ".png";

        std::string filename_pose  = path_ + "/";
        std::string file_extension_pose = ".txt";

        // Get Pose
        Eigen::Matrix<float, 3, 3, Eigen::RowMajor> erreMats = camPose.linear ();
        Eigen::Vector3f teVecs = camPose.translation ();

        // Create filenames
        filename_pose  = filename_pose  + boost::lexical_cast<std::string> (screenshot_counter) + file_extension_pose;
        filename_image = filename_image + boost::lexical_cast<std::string> (screenshot_counter) + file_extension_image;
        filename_depth = filename_depth + boost::lexical_cast<std::string> (screenshot_counter) + file_extension_image;

        // Save Image
        if ( (rgb24.cols > 0) && (rgb24.rows > 0) )
        {
            pcl::io::saveRgbPNGFile ( filename_image, (unsigned char*)rgb24.data, rgb24.cols,rgb24.rows );
        }
        else
        {
            std::cerr << "rgb24 has no data...not dumping..." << std::endl;
        }
        if ( (depth16.cols > 0) && (depth16.rows > 0) )
        {
            pcl::io::saveShortPNGFile( filename_depth, (unsigned short*)depth16.data, depth16.cols, depth16.rows, 1 );
        }
        else
        {
            std::cerr << "depth16 has no data...not dumping..." << std::endl;
        }

        // Write pose
        //writePose ( filename_pose, teVecs, erreMats );
        writePose ( filename_pose, camPose );
        writePose ( path_ + "/poses.txt", camPose, std::ios_base::out | std::ios_base::app );

        ++screenshot_counter;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void
    MyScreenshotManager::setCameraIntrinsics (float fx, float fy, float cx, float cy )
    {
        fx_ = fx;
        fy_ = fy;
        cx_ = cx;
        cy_ = cy;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    void MyScreenshotManager::writePoseHeader( std::ofstream &poseFile )
    {
        if ( poseFile.is_open() )
        {
            poseFile << "R00, R01, R02, T0, R10, R11, R12, T1, R20, R21, R22, T2, fx, fy, cx, cy, imgID" << std::endl;
        }
        else
        {
            std::cerr << "MyScreenshotManager::writePoseHeader(): posefile not open..." << std::endl;
        }
    }

    void
    MyScreenshotManager::writePose( std::string const& filename_pose, Eigen::Affine3f const& pose, std::ios_base::openmode mode )
    {
        std::ofstream poseFile;
        poseFile.open ( filename_pose.c_str(), mode );

        if ( !poseFile.is_open() )
        {
            PCL_WARN ("Unable to open/create output file for camera pose!\n");
            return;
        }

        // header
        if ( mode & std::ios_base::trunc )
            writePoseHeader( poseFile );

        // content
        for ( int y = 0; y < 4; ++y )
        {
            for ( int x = 0; x < 4; ++x )
            {
                poseFile << pose( y, x ) << ",";
            }
        }

        // intrinsics
        poseFile << fx_ << ","
                 << fy_ << ","
                 << cx_ << ","
                 << cy_ << ",";

        // imgID
        poseFile << screenshot_counter << std::endl;

        // cleanup
        poseFile.close ();
    }

    void
    MyScreenshotManager::writePose(const std::string &filename_pose, const Eigen::Vector3f &teVecs, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &erreMats)
    {
        std::ofstream poseFile;
        poseFile.open ( filename_pose.c_str() );

        if ( poseFile.is_open() )
        {
            poseFile << "TVector" << std::endl << teVecs << std::endl << std::endl
                     << "RMatrix" << std::endl << erreMats << std::endl << std::endl
                     << "Camera Intrinsics: focal height width" << std::endl << fx_ << " " << cy_ << " " << cx_ << std::endl << std::endl;
            poseFile.close ();
        }
        else
        {
            PCL_WARN ("Unable to open/create output file for camera pose!\n");
        }
    }

    void
    MyScreenshotManager::readPoses( std::string path, std::map<int,Eigen::Affine3f> &poses )
    {
        std::fstream file;
        file.open( path );
        if ( !file.is_open() )
        {
            std::cerr << "MyScreenshotManager::readPoses: could not open file at path " << path << std::endl;
            return;
        }
        poses.clear();

        std::string line;
        std::vector<std::string> words;
        while ( file )
        {
            std::getline( file, line );

            words.clear(); words.reserve(21);
            char *c_line = strdup( line.c_str() );
            char *token = strtok( c_line, "," );
            while ( token != NULL )
            {
                words.push_back( token );
                token = strtok( NULL, "," );
            }

            //for ( auto word : words )
            //    std::cout << word << std::endl;

            if ( words.size() > 20 )
            {
                Eigen::Affine3f pose;
                for ( int y = 0; y < 4; ++y )
                {
                    for ( int x = 0; x < 4; ++x )
                    {
                        pose(y,x) = atof( words[y*4+x].c_str() );
                    }
                }

                //std::vector<float> intrinsics( words.begin() + 16, words.begin() + 20 );
                poses[ atoi(words[20].c_str()) ] = pose;
            }

            if ( c_line ) { free (c_line ); c_line = NULL; }
        }

        file.close();
    }
}

#endif // PCL_SCREENSHOT_MANAGER_CPP_
