#include "PolyMeshViewer.h"

#include "AmPclUtil.h"

#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>
#include <vtkCamera.h>


namespace am
{
    PolyMeshViewer::PolyMeshViewer( int cols, int rows )
        : ProcessorWithIntrinsics(), rows_(rows), cols_(cols), inited_( false )
    {
    }

    PolyMeshViewer::PolyMeshViewer( Eigen::Matrix3f p_intrinsics, int cols, int rows )
        : ProcessorWithIntrinsics( p_intrinsics ), rows_(rows), cols_(cols), inited_( false )
    {
    }

    void
    PolyMeshViewer::initViewer( std::string title, int cols, int rows )
    {
        if ( rows > 0 ) rows_ = rows;
        if ( cols > 0 ) cols_ = cols;

        // create
        visualizer_ = pcl::visualization::PCLVisualizer::Ptr( new pcl::visualization::PCLVisualizer(title.c_str()) );

        // prepare
        visualizer_->setBackgroundColor (0, 0, 0);
        visualizer_->initCameraParameters ();
        //visualizer_->setPosition (0, 500);
        visualizer_->setSize( cols_, rows_ );
        visualizer_->setCameraClipDistances( 0.001, 10.01 );
        visualizer_->setShowFPS( false );

        // calculate fovy
        double fovy = 2.0 * atan(intrinsics_(1,2) / 2.f / intrinsics_(1,1)) * 180.0 / M_PI;

        // set fovy and aspect ratio
        vtkSmartPointer<vtkRendererCollection> rens = visualizer_->getRendererCollection();
        rens->InitTraversal ();
        vtkRenderer* renderer = NULL;
        while ( (renderer = rens->GetNextItem ()) != NULL )
        {
            // fetch camera
            vtkSmartPointer<vtkCamera> cam = renderer->GetActiveCamera ();

            // vertical view angle
            cam->SetUseHorizontalViewAngle( 0 );

            // fovy
            cam->SetViewAngle( fovy );

            // aspect
            /* vtkMatrix4x4 *projmat = */ cam->GetProjectionTransformMatrix( /* aspect (w/h): */ intrinsics_(0,2)/intrinsics_(1,2), 0.001, 10.01 );
            // THIS IS SETTING THE PROJ MATRIX USING the already set VIEWANGLE
        }

        // flip flag
        inited_ = true;
    }

    void
    PolyMeshViewer::showMesh( std::string const& path, Eigen::Affine3f const& pose )
    {
        // init pointer
        meshPtr_ = pcl::PolygonMesh::Ptr( new pcl::PolygonMesh() );

        // load mesh
        pcl::io::loadPolygonFile( path, *meshPtr_ );

        // delegate
        this->showMesh( meshPtr_, pose );
    }

    void
    PolyMeshViewer::showMesh( pcl::PolygonMesh::Ptr const& meshPtr, Eigen::Affine3f const& pose )
    {
        visualizer_->removeAllPointClouds();
        visualizer_->addPolygonMesh( *meshPtr );
        am::util::pcl::setViewerPose( *visualizer_, pose );
        visualizer_->spinOnce();
    }

    // SET,GET

    pcl::visualization::PCLVisualizer::Ptr& PolyMeshViewer::VisualizerPtr()
    {
        return visualizer_;
    }

    pcl::visualization::PCLVisualizer::Ptr const& PolyMeshViewer::VisualizerPtr() const
    {
        return visualizer_;
    }

    pcl::PolygonMesh::Ptr & PolyMeshViewer::MeshPtr()
    {
        return meshPtr_;
    }

    pcl::PolygonMesh::Ptr const& PolyMeshViewer::MeshPtr() const
    {
        return meshPtr_;
    }

} // end ns am
