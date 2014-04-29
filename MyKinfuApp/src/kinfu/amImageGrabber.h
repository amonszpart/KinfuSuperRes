#ifndef AMIMAGEGRABBER_H
#define AMIMAGEGRABBER_H

#include <vtkSmartPointer.h>
#include <vtkImageData.h>
#include <pcl/io/image_grabber.h>
#include <pcl/io/openni_camera/openni_image.h>
#include <pcl/io/openni_camera/openni_depth_image.h>
#include <pcl/point_types.h>

namespace am
{
    /*template <typename PointT>*/ class AMImageGrabber : public pcl::ImageGrabber<pcl::PointXYZRGBA>
    {
        public:
            //typedef void (sig_cb_openni_image_depth_image) (const boost::shared_ptr<openni_wrapper::Image>&, const boost::shared_ptr<openni_wrapper::DepthImage>&, float constant);
            typedef void (sig_cb_vtk_image_depth_image) (vtkSmartPointer<vtkImageData> const&, vtkSmartPointer<vtkImageData> const& );

            //using pcl::ImageGrabber::ImageGrabber();


            AMImageGrabber ( std::string    const & dir
                             , float        const   frames_per_second   = 0
                                                                          , bool         const   repeat              = false )
                : pcl::ImageGrabber<pcl::PointXYZRGBA>( dir, frames_per_second, repeat, false )
                , published_( false )
            {
                image_depth_image_signal_ = this->template createSignal<sig_cb_vtk_image_depth_image>();
            }


            virtual void start () override
            {
              if (impl_->frames_per_second_ > 0)
              {
                impl_->running_ = true;
                impl_->time_trigger_.start ();
              }
              else
              {
                    this->trigger();
              }
            }

            virtual void trigger() override;
            virtual std::string getName() const override;
            bool hasImage() { std::cout << "[" << __func__ << "]: " << "returning published " << (published_?" YES" :" NO") << std::endl; return published_; }

        protected:
            boost::signals2::signal<sig_cb_vtk_image_depth_image>* image_depth_image_signal_;
            vtkSmartPointer<vtkImageData> depth_image_;
            vtkSmartPointer<vtkImageData> rgb_image_;

            bool published_; // new image published on signal since last trigger call ("fresh" flag)

    }; // ... class amImageGrabber


} //... ns am

#endif // AMIMAGEGRABBER_H
