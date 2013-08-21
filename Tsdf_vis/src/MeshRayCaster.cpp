#include "MeshRayCaster.h"

#include "BilateralFilterCuda.hpp"
#include "ViewPointMapperCuda.h"

#include "AmPclUtil.h"

#include <pcl/octree/octree.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_subdivision.h>
#include <pcl/ros/conversions.h>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>

namespace am
{
    void MeshRayCaster::run( pcl::PolygonMesh::Ptr &meshPtr, Eigen::Affine3f const& pose, cv::Mat &depth )
    {
        const int subdivIterations = 2;
        // check input
        if ( !meshPtr.get() )
        {
            std::cerr << "MeshRayCaster::run(): mesh is empty!!!" << std::endl;
            return;
        }

        // subdivision
        {
            pcl::PolygonMesh::Ptr subdividedMeshPtr = pcl::PolygonMesh::Ptr( new pcl::PolygonMesh() );
            std::cout << "Subdividing mesh...";
            subdivideMesh( meshPtr, *subdividedMeshPtr, subdivIterations );
            std::cout << "OK" << std::endl;

            std::cout << "Copying subdivided mesh...";
            meshPtr = subdividedMeshPtr;
            std::cout << "OK" << std::endl;
        }

        // octree
        {
            mesh2Octree( *meshPtr, octree_ptr_, cloud_ptr_, /* resolution: */ 3.f/512.f/(float)subdivIterations /*.0005f*/ );
            std::vector<int> hits;
            rayCast( octree_ptr_, cloud_ptr_, pose, hits, &depth );
        }
    }

    inline Eigen::Vector3f get_next_ray( int x, int y, Eigen::Matrix3f intrinsics )
    {
        Eigen::Vector2f pnt2;
        pnt2 << (float)x, (float)y;

        return am::util::pcl::point2To3D( pnt2, intrinsics );
    }

    void MeshRayCaster::rayCast( /*  in: */ OctreePtr const& octreePtr, pcl::PointCloud<PointT>::Ptr const& cloudPtr, Eigen::Affine3f const& pose,
                                 /* out: */ std::vector<int> &p_indices, cv::Mat *p_depth )
    {
        std::cout << "Raycasting...";

        Eigen::Matrix3f rotation    = pose.rotation();
        Eigen::Vector3f translation = pose.translation();

        Eigen::Vector3f ray_start, ray_next, ray_dir;
        cv::Mat depthFC1( 960, 1280, CV_32FC1 );

        /*float *mappedCoords = new float[ depthFC1.cols * depthFC1.rows * 2 ]; //float2 matrix of normalised 3D coordinates without the homogeneous part
        ViewPointMapperCuda::runCam2World( depthFC1.cols, depthFC1.rows, mappedCoords );
        int mappedCoordsIndex = 0;*/

        p_indices.clear();
        p_indices.reserve( depthFC1.cols * depthFC1.rows );

        for ( int y = 0; y < depthFC1.rows; ++y )
        {
            for ( int x = 0; x < depthFC1.cols; ++x /*, mappedCoordsIndex += 2 */)
            {
                ray_start = translation;
                ray_next = get_next_ray( x, y, intrinsics_ );
                std::cout << ray_next.transpose() << " ";
                ray_next = (Eigen::Vector3f) { mappedCoords[mappedCoordsIndex], mappedCoords[mappedCoordsIndex+1], 1.f };
                std::cout << ray_next.transpose() << std::endl;
                ray_next = rotation * ray_next + ray_start;
                ray_dir   = static_cast<Eigen::Vector3f>(ray_next - ray_start).normalized();

                std::vector<int> pnt_indices;
                octreePtr->getIntersectedVoxelIndices( ray_start, ray_dir, pnt_indices, /* max_voxel_count: */ 1 );

                if ( pnt_indices.size() > 0 )
                {
                    p_indices.push_back( pnt_indices[0] );

                    PointT cloud_pnt3 = cloudPtr->at( pnt_indices[0] );
                    Eigen::Vector3f pnt3;
                    pnt3 << cloud_pnt3.x, cloud_pnt3.y, cloud_pnt3.z;
                    depthFC1.at<float>(y,x) = static_cast<Eigen::Vector3f>(ray_start - pnt3).norm();
                }
            }
        }
        std::cout << "OK" << std::endl;
        if ( p_depth )
        {
            depthFC1.convertTo( *p_depth, CV_16UC1, 1000.f );
        }

        // cleanup

        if ( mappedCoords ) { delete [] mappedCoords; mappedCoords = NULL; }
        return;


        {
            double minVal, maxVal;
            cv::minMaxIdx( depthFC1, &minVal, &maxVal );
            std::cout << "minVal(depth): " << minVal << ", "
                      << "maxVal(depth): " << maxVal << std::endl;
        }

        BilateralFilterCuda<float> bfc;
        bfc.setIterations( 3 );
        bfc.setFillMode( FILL_ONLY_ZEROS );
        cv::Mat bilfiltered;
        bfc.runBilateralFiltering( depthFC1, cv::Mat(), bilfiltered,
                                   5.f, .1f, 10, 1.f );
        //cv::imshow( "bilf", bilfiltered / depMax );
        bilfiltered.convertTo( depthFC1, CV_16UC1, 1000.f );
        cv::Mat dep8, diff;
        depthFC1.convertTo( dep8, CV_8UC1, 255.f / 5000.f );

        cv::imshow( "dep8", dep8 );

    }

    void MeshRayCaster::mesh2Octree( pcl::PolygonMesh &mesh, Octree::Ptr &octreePtr, pcl::PointCloud<PointT>::Ptr &cloudPtr, float resolution )
    {
        std::cout << "mesh2Octree...";

        cloudPtr = pcl::PointCloud<PointT>::Ptr( new pcl::PointCloud<PointT> );

        //Input of the above cloud and the corresponding output of cloud_pcl
        pcl::fromPCLPointCloud2( mesh.cloud, *cloudPtr ); //pcl::fromROSMsg( mesh.cloud, *cloud_pcl );

        octreePtr = OctreePtr( new Octree(resolution) );
        octreePtr->setInputCloud( cloudPtr );
        octreePtr->defineBoundingBox();
        octreePtr->addPointsFromInputCloud();

        std::cout << "Octree Leaf Count Operation: " << octreePtr->getLeafCount() << std::endl;
        std::cout << "Octree Branch Count Operation: " << octreePtr->getBranchCount() << std::endl;

        std::cout << "OK" << std::endl;
    }

    void MeshRayCaster::subdivideMesh( pcl::PolygonMesh::ConstPtr input_mesh, pcl::PolygonMesh &output_mesh, int iterations )
    {
        // initialize
        pcl::MeshSubdivisionVTK msvtk;
        msvtk.setFilterType( pcl::MeshSubdivisionVTK::LINEAR );

        // setup pingpong
        pcl::PolygonMesh::Ptr tmp[2];
        tmp[0] = pcl::PolygonMeshPtr( new pcl::PolygonMesh );
        tmp[1] = pcl::PolygonMeshPtr( new pcl::PolygonMesh );
        int mesh_id = 0;

        // iterate
        for ( int it = 0; it < iterations - 1; ++it )
        {
            // set input
            msvtk.setInputMesh( !it ? input_mesh : tmp[mesh_id] );
            // process
            msvtk.process     ( *tmp[(mesh_id+1)%2] );
            // pingpong
            mesh_id = (mesh_id + 1) % 2;
        }
        // last iteration
        msvtk.setInputMesh( (iterations > 1) ? tmp[mesh_id] : input_mesh );
        msvtk.process( output_mesh );
    }

} // end ns am
