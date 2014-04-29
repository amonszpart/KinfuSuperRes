#include "amImageGrabber.h"


//    template <typename PointT>
void am::AMImageGrabber::trigger()
{
    published_ = false;
    if ( this->impl_->valid_ )
    {
        std::cout << "[" << __func__ << "]: " << "image_depth_image_signal_..."; fflush(stdout);

        image_depth_image_signal_->operator ()( rgb_image_, depth_image_ );
        this->impl_->valid_ = false;
        published_          = true;

        std::cout << "[" << __func__ << "]: " << "ok...\n"; fflush(stdout);
    }

    /// load next
    {
        if ( this->impl_->cur_frame_ >= this->impl_->numFrames() )
        {
            if ( this->impl_->repeat_ )  this->impl_->cur_frame_ = 0;
            else
            {
                this->impl_->valid_ = false;
                return;
            }
        }
        std::string const& depth_image_file = this->impl_->depth_image_files_[this->impl_->cur_frame_];
        // If there are RGB files, load an rgb image
        if ( this->impl_->rgb_image_files_.size() )
        {
            std::string const& rgb_image_file = this->impl_->rgb_image_files_[this->impl_->cur_frame_];
            // If we were unable to pull a Vtk image, throw an error
            std::cout << "[" << __func__ << "]: " << "getting " << rgb_image_file << "...";
            if ( !this->impl_->getVtkImage(rgb_image_file, rgb_image_) )
            {
                std::cerr << "[" << __func__ << "]: " << "could not read rgb_image_files_[" << this->impl_->cur_frame_ << "]: " << rgb_image_file << std::endl;
                this->impl_->valid_ = false;
                return;
            }
            std::cout << "ok...\n";
        }

        std::cout << "[" << __func__ << "]: " << "getting " << depth_image_file << "...";
        if ( !this->impl_->getVtkImage(depth_image_file, depth_image_) )
        {
            std::cerr << "[" << __func__ << "]: " << "could not read depth_image_files_[" << this->impl_->cur_frame_ << "]: " << depth_image_file << std::endl;
            this->impl_->valid_ = false;
            return;
        }
        std::cout << "ok...\n";

        ++(this->impl_->cur_frame_);
        this->impl_->valid_ = true;
        std::cout << "this->impl_->cur_frame_: " << this->impl_->cur_frame_ << ", is_valid: " << (this->impl_->valid_?" YES": " NO") << std::endl;
    }
}

//    template <typename PointT>
std::string am::AMImageGrabber::getName() const
{
    return ("AMImageGrabber");
}

#if 0
template <typename PointT> void am::AMImageGrabber<PointT>::trigger()
{
    if ( this->impl_->valid_ )
    {
        image_depth_image_signal_->operator ()( rgb_image_, depth_image_ );
    }

    /// load next
    {
        if ( this->impl_->cur_frame_ >= this->impl_->numFrames() )
        {
            if ( this->impl_->repeat_ )  this->impl_->cur_frame_ = 0;
            else
            {
                this->impl_->valid_ = false;
                return;
            }
        }
        std::string const& depth_image_file = this->impl_->depth_image_files_[this->impl_->cur_frame];
        // If there are RGB files, load an rgb image
        if ( this->impl->rgb_image_files_.size() )
        {
            std::string const& rgb_image_file = this->impl_->rgb_image_files_[this->impl_->cur_frame];
            // If we were unable to pull a Vtk image, throw an error
            if ( !this->impl_->getVtkImage(rgb_image_file, rgb_image_) )
            {
                std::cerr << "[" << __func__ << "]: " << "could not read rgb_image_files_[" << this->impl_->cur_frame << "]: " << rgb_image_file << std::endl;
                this->impl_->valid_ = false;
                return;
            }
        }

        if ( !this->impl_->getVtkImage(depth_image_file, depth_image_) )
        {
            std::cerr << "[" << __func__ << "]: " << "could not read depth_image_files_[" << this->impl_->cur_frame << "]: " << depth_image_file << std::endl;
            this->impl_->valid_ = false;
            return;
        }

        ++(this->impl_->cur_frame_);
    }
}
#endif
