#ifndef __POINT_CLOUD_COLOR_HANDLER_RGB_CLOUD
#define __POINT_CLOUD_COLOR_HANDLER_RGB_CLOUD

#include "kinfu_pcl_headers.h"

namespace pcl {

    namespace visualization
    {
        //////////////////////////////////////////////////////////////////////////////////////
        /** \brief RGB handler class for colors. Uses the data present in the "rgb" or "rgba"
        * fields from an additional cloud as the color at each point.
        * \author Anatoly Baksheev
        * \ingroup visualization
        */
        template <typename PointT>
        class PointCloudColorHandlerRGBCloud : public PointCloudColorHandler<PointT>
        {
                using PointCloudColorHandler<PointT>::capable_;
                using PointCloudColorHandler<PointT>::cloud_;

                typedef typename PointCloudColorHandler<PointT>::PointCloud::ConstPtr PointCloudConstPtr;
                typedef typename pcl::PointCloud<RGB>::ConstPtr RgbCloudConstPtr;

            public:
                typedef boost::shared_ptr<PointCloudColorHandlerRGBCloud<PointT> > Ptr;
                typedef boost::shared_ptr<const PointCloudColorHandlerRGBCloud<PointT> > ConstPtr;

                /** \brief Constructor. */
                PointCloudColorHandlerRGBCloud (const PointCloudConstPtr& cloud, const RgbCloudConstPtr& colors)
                    : rgb_ (colors)
                {
                    cloud_  = cloud;
                    capable_ = true;
                }

                /** \brief Obtain the actual color for the input dataset as vtk scalars.
      * \param[out] scalars the output scalars containing the color for the dataset
      * \return true if the operation was successful (the handler is capable and
      * the input cloud was given as a valid pointer), false otherwise
      */
                virtual bool
                getColor (vtkSmartPointer<vtkDataArray> &scalars) const
                {
                    if (!capable_ || !cloud_)
                        return (false);

                    if (!scalars)
                        scalars = vtkSmartPointer<vtkUnsignedCharArray>::New ();
                    scalars->SetNumberOfComponents (3);

                    vtkIdType nr_points = vtkIdType (cloud_->points.size ());
                    reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->SetNumberOfTuples (nr_points);
                    unsigned char* colors = reinterpret_cast<vtkUnsignedCharArray*>(&(*scalars))->GetPointer (0);

                    // Color every point
                    if (nr_points != int (rgb_->points.size ()))
                        std::fill (colors, colors + nr_points * 3, static_cast<unsigned char> (0xFF));
                    else
                        for (vtkIdType cp = 0; cp < nr_points; ++cp)
                        {
                            int idx = cp * 3;
                            colors[idx + 0] = rgb_->points[cp].r;
                            colors[idx + 1] = rgb_->points[cp].g;
                            colors[idx + 2] = rgb_->points[cp].b;
                        }
                    return (true);
                }

            private:
                virtual std::string
                getFieldName () const { return ("additional rgb"); }
                virtual std::string
                getName () const { return ("PointCloudColorHandlerRGBCloud"); }

                RgbCloudConstPtr rgb_;
        };
    }
}

#endif
