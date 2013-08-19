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
 *  Author: Francisco, Technical University Eindhoven, (f.j.mysurname.soriano <At > tue.nl)
 */

#ifndef MY_SCREENSHOT_MANAGER_H_
#define MY_SCREENSHOT_MANAGER_H_

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

//#include "kinfu_pcl_headers.h"
//#include <boost/filesystem.hpp>
//#include <pcl/gpu/kinfu/kinfu.h>

//#include <pcl/gpu/kinfu/kinfu.h>
#include "kinfu.h"
#include <pcl/pcl_exports.h>
#include <pcl/gpu/containers/device_array.h>
#include <pcl/gpu/containers/kernel_containers.h>
#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp> 
#include <boost/filesystem/path.hpp>
//#include <boost/graph/buffer_concepts.hpp>
#include <pcl/console/print.h>
#include <map>

using namespace pcl::gpu;

namespace am
{
    /** \brief Screenshot Manager saves a screenshot with the corresponding camera pose from Kinfu. Please create a folder named "KinFuSnapshots" in the folder where you call kinfu.
        * \author Francisco Heredia
        */
    class MyScreenshotManager
    {
        public:

            //typedef pcl::gpu::kinfuLS::PixelRGB PixelRGB;

            /** Constructor */
            MyScreenshotManager();

            /** Destructor */
            ~MyScreenshotManager(){}

            /** \brief Sets Depth camera intrinsics
            * \param[in] fx focal length x
            * \param[in] fy focal length y
            * \param[in] cx principal point x
            * \param[in] cy principal point y
            */
            void
            setCameraIntrinsics (float fx = 575.816f, float fy = 575.816f, float cx = 640.0f, float cy = 480.0f );

            void
            setPath( std::string const& path ) { path_ = path; }

            /**Save Screenshot*/
            void
            saveImage( const Eigen::Affine3f &camPose, const pcl::gpu::PtrStepSz<const pcl::gpu::PixelRGB> &rgb24, const pcl::gpu::PtrStepSz<const unsigned short>& depth16 );

            static void
            readPoses( std::string path, std::map<int,Eigen::Affine3f> &poses );

        protected:
            std::string path_;

            /** Write camera pose to file */
            void
            writePose( std::string const& filename_pose, Eigen::Affine3f const& pose, std::ios_base::openmode mode = std::ios_base::out | std::ios_base::trunc );
            void
            writePoseHeader( std::ofstream &poseFile );

            void
            writePose(const std::string &filename_pose, const Eigen::Vector3f &teVecs, const Eigen::Matrix<float, 3, 3, Eigen::RowMajor> &erreMats);




            /**Counter of the number of screenshots taken*/
            int screenshot_counter;

            /** \brief Intrinsic parameters of depth camera. */
            float fx_, fy_, cy_, cx_;
    };

}

#endif // PCL_SCREENSHOT_MANAGER_H_
