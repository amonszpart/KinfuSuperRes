#ifndef IMAGE_VIEW_H
#define IMAGE_VIEW_H

#include "kinfu_pcl_headers.h"

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;
namespace pc = pcl::console;

namespace pcl
{
    namespace gpu
    {
        void paint3DView (const KinfuTracker::View& rgb24, KinfuTracker::View& view, float colors_weight = 0.5f);
    }
}

struct ImageView
{
        ImageView ( int viz );

        void
        showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, Eigen::Affine3f* pose_ptr = 0);

        void
        showDepth (const PtrStepSz<const unsigned short>& depth);
        void
        showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose);
        void
        toggleImagePaint();

        int viz_;
        bool paint_image_;
        bool accumulate_views_;

        visualization::ImageViewer::Ptr viewerScene_;
        visualization::ImageViewer::Ptr viewerDepth_;
        //visualization::ImageViewer viewerColor_;

        KinfuTracker::View view_device_;
        KinfuTracker::View colors_device_;
        vector<KinfuTracker::PixelRGB> view_host_;

        RayCaster::Ptr raycaster_ptr_;

        KinfuTracker::DepthMap generated_depth_;

#ifdef HAVE_OPENCV
        vector<cv::Mat> views_;
#endif
};

#endif // IMAGE_VIEW_H
