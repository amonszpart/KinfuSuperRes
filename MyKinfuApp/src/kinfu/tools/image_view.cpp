#include "image_view.h"

ImageView::ImageView(int viz)
    : viz_(viz), paint_image_ (true), accumulate_views_ (false)
{
    if (viz_)
    {
        viewerScene_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);
        viewerDepth_ = pcl::visualization::ImageViewer::Ptr(new pcl::visualization::ImageViewer);

        viewerScene_->setWindowTitle ("View3D from ray tracing");
        viewerScene_->setPosition (0, 0);
        viewerDepth_->setWindowTitle ("Kinect Depth stream");
        viewerDepth_->setPosition (640, 0);
        //viewerColor_.setWindowTitle ("Kinect RGB stream");
    }
}

void
ImageView::showScene (KinfuTracker& kinfu, const PtrStepSz<const KinfuTracker::PixelRGB>& rgb24, bool registration, Eigen::Affine3f* pose_ptr )
{
    if (pose_ptr)
    {
        raycaster_ptr_->run(kinfu.volume(), *pose_ptr);
        raycaster_ptr_->generateSceneView(view_device_);
    }
    else
        kinfu.getImage (view_device_);

    if (paint_image_ && registration && !pose_ptr)
    {
        colors_device_.upload (rgb24.data, rgb24.step, rgb24.rows, rgb24.cols);
        paint3DView (colors_device_, view_device_);
    }


    int cols;
    view_device_.download (view_host_, cols);
    if (viz_)
        viewerScene_->showRGBImage (reinterpret_cast<unsigned char*> (&view_host_[0]), view_device_.cols (), view_device_.rows ());

    //viewerColor_.showRGBImage ((unsigned char*)&rgb24.data, rgb24.cols, rgb24.rows);

#ifdef HAVE_OPENCV
    if (accumulate_views_)
    {
        views_.push_back (cv::Mat ());
        cv::cvtColor (cv::Mat (480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back (), CV_RGB2GRAY);
        //cv::copy(cv::Mat(480, 640, CV_8UC3, (void*)&view_host_[0]), views_.back());
    }
#endif
}

void
ImageView::showDepth (const PtrStepSz<const unsigned short>& depth)
{
    if (viz_)
        viewerDepth_->showShortImage (depth.data, depth.cols, depth.rows, 0, 5000, true);
}

void
ImageView::showGeneratedDepth (KinfuTracker& kinfu, const Eigen::Affine3f& pose)
{
    raycaster_ptr_->run(kinfu.volume(), pose);
    raycaster_ptr_->generateDepthImage(generated_depth_);

    int c;
    vector<unsigned short> data;
    generated_depth_.download(data, c);

    if (viz_)
        viewerDepth_->showShortImage (&data[0], generated_depth_.cols(), generated_depth_.rows(), 0, 5000, true);
}

void
ImageView::toggleImagePaint()
{
    paint_image_ = !paint_image_;
    cout << "Paint image: " << (paint_image_ ? "On   (requires registration mode)" : "Off") << endl;
}

