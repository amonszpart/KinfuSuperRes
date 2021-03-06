#include "MeshRayCaster.h"

#include "BilateralFilterCuda.hpp"
#include "ViewPointMapperCuda.h"

#include "AmPclUtil.h"

#include <pcl/octree/octree.h>
#include <pcl/surface/vtk_smoothing/vtk_mesh_subdivision.h>
#include <pcl/ros/conversions.h>
#include <pcl/surface/vtk_smoothing/vtk_utils.h>
#include <vtkApproximatingSubdivisionFilter.h>
#include <vtkCatmullClarkFilter.h>
#include <vtkSmartPointer.h>

#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/core/core.hpp>

namespace am
{
    void
    MeshRayCaster::run( /* out: */ cv::Mat &depth,
                        /*  in: */ pcl::PolygonMesh::Ptr &meshPtr, Eigen::Affine3f const& p_pose, int subdivideIterations )
    {
        // check input
        if ( !meshPtr.get() )
        {
            std::cerr << "MeshRayCaster::run(): mesh is empty!!!" << std::endl;
            return;
        }

        // subdivision
        if ( subdivideIterations > 0 )
        {
            std::cout << "Subdividing mesh...";

            pcl::PolygonMesh::Ptr subdividedMeshPtr = pcl::PolygonMesh::Ptr( new pcl::PolygonMesh() );
            subdivideMesh( *subdividedMeshPtr, meshPtr, subdivideIterations );
            meshPtr = subdividedMeshPtr;

            std::cout << "OK" << std::endl;
        }

        // octree
        {
            mesh2Octree( /*  outOctree: */ octree_ptr_,
                         /*   outCloud: */ cloud_ptr_,
                         /*     inMesh: */ meshPtr,
                         /* resolution: */ 3.f / 512.f / (float)subdivideIterations ); // .0005f
        }

        // rayCast
        {
            std::vector<int> hits; //unused
            rayCast( hits, &depth, octree_ptr_, cloud_ptr_, p_pose );
        }
    }

    void
    MeshRayCaster::enhanceMesh( /* out: */ pcl::PolygonMesh::Ptr &outMeshPtr,
                                /*  in: */ cv::Mat const& dep16, pcl::PolygonMesh::ConstPtr const& inMeshPtr, Eigen::Affine3f const& p_pose,
                                const float resolution )
    {
        // start
        std::cout << "MeshRayCaster::enhanceMesh() starting..." << std::endl;

        // check input
        if ( !inMeshPtr.get() )
        {
            std::cerr << "MeshRayCaster::enhanceMesh(): inMeshPtr empty!" << std::endl;
            return;
        }

        // init output
        outMeshPtr.reset( new pcl::PolygonMesh );
        *outMeshPtr = *inMeshPtr;
        {
            pcl::PointCloud<pcl::PointXYZ> tmpCloud;
            pcl::fromROSMsg( inMeshPtr->cloud, tmpCloud );
            pcl::PointCloud<pcl::PointXYZRGB> cloud2;
            //pcl::PolygonMesh mesh2;
            //std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> >::iterator it1;
            pcl::PointXYZRGB tmp;

            for ( auto it1 = tmpCloud.points.begin() ; it1 != tmpCloud.points.end() ; ++it1 )
            {
                tmp.x = it1->x;
                tmp.y = it1->y;
                tmp.z = it1->z;
                tmp.r = 127;
                tmp.g = 128;
                tmp.b = 129;
                cloud2.push_back(tmp);
            }
            pcl::toROSMsg( cloud2, outMeshPtr->cloud );
        }

        std::cout << "EnhanceMesh: inMeshPtr->cloud.size: " << inMeshPtr->cloud.width << "x" <<  inMeshPtr->cloud.height << std::endl;
        std::cout << "EnhanceMesh: outMeshPtr->cloud.size: " << outMeshPtr->cloud.width << "x" <<  outMeshPtr->cloud.height << std::endl;
        const int point_step = outMeshPtr->cloud.point_step;
        const int x_offs     = outMeshPtr->cloud.fields[0].offset;
        const int y_offs     = outMeshPtr->cloud.fields[1].offset;
        const int z_offs     = outMeshPtr->cloud.fields[2].offset;
        const int rgb_offs     = outMeshPtr->cloud.fields[3].offset;

        // octree
        Octree::Ptr octreePtr;
        pcl::PointCloud<PointT>::Ptr cloudPtr;
        {
            mesh2Octree( /* out    Octree: */ octreePtr,
                         /* out     Cloud: */ cloudPtr,
                         /* in       Mesh: */ inMeshPtr,
                         /* in resolution: */ resolution ); // .0005f
        }

        // work
        Eigen::Matrix3f rotation    = p_pose.rotation();
        Eigen::Vector3f translation = p_pose.translation();
        Eigen::Vector3f pnt3D;

        const int HIST_SIZE = 100;
        std::vector<int> hist( HIST_SIZE );
        for ( int y = 0; y < dep16.rows; ++y )
        {
            for ( int x = 0; x < dep16.cols; ++x )
            {
                pnt3D = rotation *
                        ( am::util::pcl::point2To3D((Eigen::Vector2f){x,y}, intrinsics_) *
                          (float)dep16.at<ushort>(y,x) / 1000.f )
                        + translation;
                PointT pclPnt( pnt3D(0), pnt3D(1), pnt3D(2) );

                std::vector<int>   k_indices;
                std::vector<float> k_sqr_distances;
                int max_nn = 5;
                octreePtr->radiusSearch( pclPnt, resolution*2.f, k_indices, k_sqr_distances, max_nn );

                if ( k_indices.size() > 0 )
                {
                    for ( int k = 0; k < std::min(5,(int)k_indices.size()); ++k )
                    {
                        int pnt_idx = k_indices[k];
                        float *p_x = reinterpret_cast<float*>( &(outMeshPtr->cloud.data[pnt_idx * point_step + x_offs]) );
                        float *p_y = reinterpret_cast<float*>( &(outMeshPtr->cloud.data[pnt_idx * point_step + y_offs]) );
                        float *p_z = reinterpret_cast<float*>( &(outMeshPtr->cloud.data[pnt_idx * point_step + z_offs]) );

                        uchar *p_bgr = reinterpret_cast<uchar*>( &(outMeshPtr->cloud.data[pnt_idx * point_step + rgb_offs]) );
                        /*std::cout << "p_rgb: "
                                  << (int)p_bgr[0] << " "
                                  << (int)p_bgr[1] << " "
                                  << (int)p_bgr[2]
                                  << std::endl;*/

                        /*std::cout << "substituting: "
                                  << *p_x << "," << *p_y << "," << *p_z
                                  << " to "
                                  << pclPnt.x << "," << pclPnt.y << "," << pclPnt.z << std::endl;*/

                        float dx = (*p_x - pclPnt.x);
                        float dy = (*p_y - pclPnt.y);
                        float dz = (*p_z - pclPnt.z);
                        p_bgr[2] = 128 + std::min((uchar)127,(uchar)round(sqrt(dx*dx+dy*dy+dz*dz) * 100.f));
                        p_bgr[1] = 255;
                        std::cout << round(sqrt(dx*dx+dy*dy+dz*dz)) << " ";

                        *p_x = pclPnt.x;
                        *p_y = pclPnt.y;
                        *p_z = pclPnt.z;
                    }
                }

                // update histogram
                ++hist[ std::min((int)k_indices.size(),HIST_SIZE-1) ];
            }
        }

        // plot hist
        std::cout << "/nhist: ";
        for ( int i = 0; i < hist.size(); ++i )
        {
            if ( hist[i] > 0 ) std::cout << i << ": " << hist[i] << "; ";
        }
        std::cout << std::endl;


        // finish
        std::cout << "MeshRayCaster::enhanceMesh() finished..." << std::endl;
    }

    void addFace( pcl::PolygonMesh::Ptr &meshPtr, std::vector<Eigen::Vector3f> points, std::vector<Eigen::Vector3f> *colors )
    {
        int vxId = meshPtr->cloud.width;
        //std::cout << "vxid: " << vxId << std::endl;
        meshPtr->cloud.width += 3;
        meshPtr->cloud.data.resize( meshPtr->cloud.width * meshPtr->cloud.point_step );
        float* tmp;
        ::pcl::Vertices face;
        for ( int pid = 0; pid < 3; ++pid, ++vxId )
        {
            face.vertices.push_back( vxId );
            for ( int i = 0; i < 3; ++i )
            {
                tmp = reinterpret_cast<float*>( &(meshPtr->cloud.data[vxId * meshPtr->cloud.point_step + meshPtr->cloud.fields[i].offset]) );
                *tmp = points[pid](i);
            }
            if ( colors )
            {
                tmp = reinterpret_cast<float*>( &(meshPtr->cloud.data[ vxId * meshPtr->cloud.point_step + meshPtr->cloud.fields[3].offset]) );
                for ( int i = 0; i < 3; ++i )
                {
                    tmp[i] = colors->at(pid)(i);
                }
            }
        }

        meshPtr->polygons.push_back( face );
        meshPtr->cloud.row_step = meshPtr->cloud.point_step * meshPtr->cloud.width;
    }

    void
    MeshRayCaster::enhanceMesh2( /* out: */ pcl::PolygonMesh::Ptr &outMeshPtr,
                                  /*  in: */ cv::Mat const& dep16, pcl::PolygonMesh::ConstPtr const& inMeshPtr, Eigen::Affine3f const& p_pose,
                                  std::vector<cv::Mat> depths, std::vector<cv::Mat> indices )
    {
        if ( dep16.type() != CV_32FC1 )
        {
            std::cerr << "MeshRaycaster::enhanceMesh2(): wrong dep type...need float" << std::endl;
        }
        // start
        std::cout << "MeshRayCaster::enhanceMesh2() starting..." << std::endl;

        // check input
        if ( !inMeshPtr.get() )
        {
            std::cerr << "MeshRayCaster::enhanceMesh2(): inMeshPtr empty!" << std::endl;
            return;
        }
        // copy // TODO: add colour
        *outMeshPtr = *inMeshPtr;
        float maxx,maxy, maxz; maxx = maxy = maxz = 0.f;
        {
            pcl::PointCloud<pcl::PointXYZ> tmpCloud;
            pcl::fromROSMsg( inMeshPtr->cloud, tmpCloud );
            pcl::PointCloud<pcl::PointXYZRGB> cloud2;
            //pcl::PolygonMesh mesh2;
            //std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ> >::iterator it1;
            pcl::PointXYZRGB tmp;

            for ( auto it1 = tmpCloud.points.begin() ; it1 != tmpCloud.points.end() ; ++it1 )
            {
                tmp.x = it1->x; if ( tmp.x > maxx ) maxx = tmp.x;
                tmp.y = it1->y; if ( tmp.y > maxy ) maxy = tmp.y;
                tmp.z = it1->z; if ( tmp.z > maxz ) maxz = tmp.z;

                tmp.r = 127;
                tmp.g = 25;
                tmp.b = 129;
                cloud2.push_back(tmp);
            }
            pcl::toROSMsg( cloud2, outMeshPtr->cloud );
        }
        std::cout << "maxx: " << maxx
                  << " maxy: " << maxy
                  << " maxz: " << maxz << std::endl;

        const int point_step = outMeshPtr->cloud.point_step;
        const int x_offs     = outMeshPtr->cloud.fields[0].offset;
        const int y_offs     = outMeshPtr->cloud.fields[1].offset;
        const int z_offs     = outMeshPtr->cloud.fields[2].offset;
        const int rgb_offs   = outMeshPtr->cloud.fields[3].offset;
        const Eigen::Matrix3f rotation    = p_pose.rotation();
        const Eigen::Vector3f translation = p_pose.translation();

#if 0
        addFace( outMeshPtr,
                    (std::vector<Eigen::Vector3f>){
                        (Eigen::Vector3f){1.f, 1.f, 2.f},
                        (Eigen::Vector3f){1.f, 5.f, 2.f},
                        (Eigen::Vector3f){5.f, 5.f, 2.f}
                    }, NULL);
        outMeshPtr->polygons.resize( outMeshPtr->polygons.size()+1 );
        outMeshPtr->polygons.back().vertices.resize( 3 );
#endif

#if 0
        int vxid = outMeshPtr->cloud.width-3;
        // add place for 3 new vertices
        //outMeshPtr->cloud.width += 3;
        //outMeshPtr->cloud.data.resize( outMeshPtr->cloud.width * point_step );

        // prepare face
        ::pcl::Vertices newFace;

        // add 3 vertices
        for ( int oid = 0; oid < 3; ++oid, ++vxid )
        {
            // fill face
            newFace.vertices.push_back( vxid );

            // add point
            float *p_x = reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + x_offs]) );
            std::cout << "writing to " << vxid * point_step + x_offs << " value: " << oid * 2.f
                      << "replacing " << (*p_x)<< " so: ";
            *p_x = oid * 2.f;
            std::cout << *p_x << std::endl;

            float *p_y = reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + y_offs]) );
            *p_y = (float)oid * 2.f;
            std::cout << "writing to " <<  vxid * point_step + y_offs << " value: " << oid * 2.f << std::endl;
            float *p_z = reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + z_offs]) );
            *p_z = 4.f - (float)oid *  2.f;
            std::cout << "writing to " <<  vxid * point_step + z_offs << " value: " << 4.f - oid *  2.f << std::endl;
            *reinterpret_cast<uchar*>( &(outMeshPtr->cloud.data[ vxid * point_step + rgb_offs]) ) = 255;
        }

        // add face
        outMeshPtr->polygons.resize(outMeshPtr->polygons.size()+1);
        outMeshPtr->polygons.back().vertices.resize(3);
        outMeshPtr->polygons.back().vertices[0] = newFace.vertices[0];
        outMeshPtr->polygons.back().vertices[1] = newFace.vertices[1];
        outMeshPtr->polygons.back().vertices[2] = newFace.vertices[2];
        outMeshPtr->polygons.resize(outMeshPtr->polygons.size()+1);
        outMeshPtr->polygons.back().vertices.resize(3);
        //outMeshPtr->cloud.row_step = outMeshPtr->cloud.width * outMeshPtr->cloud.point_step;
#endif

        // clear mesh (we only needed the header from inMeshPtr)
        outMeshPtr->cloud.data.clear();
        outMeshPtr->cloud.width = 0;
        outMeshPtr->cloud.height = 1;
        outMeshPtr->polygons.clear();

        // triangle index offsets
        std::vector< std::vector<cv::Vec2i> > smalls(2);
        {
            smalls[0].push_back( cv::Vec2i( 0, 0) );
            smalls[0].push_back( cv::Vec2i(-1, 0) );
            smalls[0].push_back( cv::Vec2i( 0,-1) );

            smalls[1].push_back( cv::Vec2i( 0, 0) );
            smalls[1].push_back( cv::Vec2i(-1, 1) );
            smalls[1].push_back( cv::Vec2i(-1, 0) );
        }
#if 1
        // variables
        Eigen::Vector3f pnt3D;
        std::vector<bool> keepFaces( inMeshPtr->polygons.size() );
        std::fill( keepFaces.begin(), keepFaces.end(), true );
        std::vector< ::pcl::Vertices> newFaces;

        // indices[1] contains rounded triangleIDs as unsigned fields
        for ( int y = 0; y < indices[1].rows; ++y )
        {
            for ( int x = 0; x < indices[1].cols; ++x )
            {
                // for every triangle in the image grid ( /_| and  T/ )
                for ( int smid = 0; smid < smalls.size(); ++smid )
                {
                    std::vector<cv::Vec2i> offs = smalls[ smid ];

                    if (
                         (indices[1].at<unsigned>(y+offs[0][0],x+offs[0][1]) != 0)
                         //&& indices[1].at<unsigned>(y+offs[0][0],x+offs[0][1]) == indices[1].at<unsigned>(y+offs[1][0],x+offs[1][1])
                         //&& indices[1].at<unsigned>(y+offs[0][0],x+offs[0][1]) == indices[1].at<unsigned>(y+offs[2][0],x+offs[2][1])
                        )
                    {
                        // exclude old face
                        keepFaces[ indices[1].at<unsigned>(y,x) ] = false;


                        /*Eigen::Vector2f curr;
                        curr(0) = x + offs[oid][1];
                        curr(1) = y + offs[oid][0];
                        pnt3D = rotation * ( am::util::pcl::point2To3D(curr, intrinsics_) * (float)dep16.at<ushort>(y,x) ) + translation;
                        */
                        static bool printed = false;
                        if ( !printed  && y > 200 && x > 200 )
                        {
                            std::cout << "dep16: " << (float)dep16.at<float>(y,x) << std::endl;
                            printed = true;
                        }
                        addFace( outMeshPtr,
                                 (std::vector<Eigen::Vector3f>){
                                     rotation * ( am::util::pcl::point2To3D((Eigen::Vector2f){x + offs[0][1],y + offs[0][0]}, intrinsics_) * (float)dep16.at<float>(y+offs[0][0],x+offs[0][1]) ) + translation,
                                     rotation * ( am::util::pcl::point2To3D((Eigen::Vector2f){x + offs[1][1],y + offs[1][0]}, intrinsics_) * (float)dep16.at<float>(y+offs[1][0],x+offs[1][1]) ) + translation,
                                     rotation * ( am::util::pcl::point2To3D((Eigen::Vector2f){x + offs[2][1],y + offs[2][0]}, intrinsics_) * (float)dep16.at<float>(x+offs[2][0],x+offs[2][1]) ) + translation,
                                 }, NULL);
#if 0
                        // remember vertexid to insert at
                        int vxid = outMeshPtr->cloud.width;
                        // add place for 3 new vertices
                        outMeshPtr->cloud.width += 3;
                        outMeshPtr->cloud.data.resize( outMeshPtr->cloud.width * point_step );

                        // prepare face
                        ::pcl::Vertices newFace;

                        // add 3 vertices
                        for ( int oid = 0; oid < offs.size(); ++oid, ++vxid )
                        {
                            // fill face
                            newFace.vertices.push_back( vxid );

                            // calculate point
                            Eigen::Vector2f curr;
                            curr(0) = x + offs[oid][1];
                            curr(1) = y + offs[oid][0];
                            pnt3D = rotation * ( am::util::pcl::point2To3D(curr, intrinsics_) * (float)dep16.at<ushort>(y,x) ) + translation;

                            // add point
                            *reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + x_offs]) ) = pnt3D(0);
                            *reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + y_offs]) ) = pnt3D(1);
                            *reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + z_offs]) ) = pnt3D(2);
                            *reinterpret_cast<uchar*>( &(outMeshPtr->cloud.data[ vxid * point_step + rgb_offs]) ) = 255;
                        }

                        // add face
                        outMeshPtr->polygons.push_back( newFace );
#endif
                    }
                }
            }
        }
#endif
#if 0
        // put back kept vertices and faces
        int keptFaces = 0;
        int skip = 0;
        for ( int i = 0; i < inMeshPtr->polygons.size(); ++i )
        {
            if ( keepFaces[i] ) //FIXME start face id-s from 1, reserve 0 for empty in frag and mesh.cpp
            {
                ++keptFaces;
                // copy face
                outMeshPtr->polygons.push_back( inMeshPtr->polygons[i] );

                // copy vertices
                int vxid = outMeshPtr->cloud.width;
                outMeshPtr->cloud.width += 3;
                outMeshPtr->cloud.data.resize( outMeshPtr->cloud.width * point_step );

                for ( int polygPntId = 0; polygPntId < outMeshPtr->polygons.back().vertices.size(); ++polygPntId, ++vxid )
                {
                    int oldVxId = outMeshPtr->polygons.back().vertices[ polygPntId ];

                    *reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + x_offs]) )
                            = *reinterpret_cast<const float*>( &(inMeshPtr->cloud.data[ oldVxId * point_step + x_offs]) );
                    *reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + y_offs]) )
                            = *reinterpret_cast<const float*>( &(inMeshPtr->cloud.data[ oldVxId * point_step + y_offs]) );
                    *reinterpret_cast<float*>( &(outMeshPtr->cloud.data[ vxid * point_step + z_offs]) )
                            = *reinterpret_cast<const float*>( &(inMeshPtr->cloud.data[ oldVxId * point_step + z_offs]) );
                }
            }
            else
                skip++;
        }
        std::cout << "enhanceMesh2: keptFaces: " << keptFaces << "skip: " << skip << std::endl;
#endif
        //outMeshPtr->cloud.row_step = outMeshPtr->cloud.width * outMeshPtr->cloud.point_step;


        // finish
        std::cout << "MeshRayCaster::enhanceMesh2() finished..." << std::endl;
    }

    inline
    Eigen::Vector3f get_next_ray( int x, int y, Eigen::Matrix3f intrinsics )
    {
        Eigen::Vector2f pnt2;
        pnt2 << (float)x, (float)y;

        return am::util::pcl::point2To3D( pnt2, intrinsics );
    }

    void
    MeshRayCaster::rayCast( /* out: */ std::vector<int> &p_indices, cv::Mat *p_depth,
                            /*  in: */ OctreePtr const& octreePtr, pcl::PointCloud<PointT>::Ptr const& cloudPtr, Eigen::Affine3f const& pose )
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
            std::cout << "rayCast.y: " << y << std::endl;
            for ( int x = 0; x < depthFC1.cols; ++x /*, mappedCoordsIndex += 2 */)
            {
                ray_start = translation;
                ray_next = get_next_ray( x, y, intrinsics_ );
                //std::cout << ray_next.transpose() << " ";
                //ray_next = (Eigen::Vector3f) { mappedCoords[mappedCoordsIndex], mappedCoords[mappedCoordsIndex+1], 1.f }; (TODO: recalibrate for larger resolution)
                //std::cout << ray_next.transpose() << std::endl;
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

        //if ( mappedCoords ) { delete [] mappedCoords; mappedCoords = NULL; }
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

    void
    MeshRayCaster::mesh2Octree(Octree::Ptr &octreePtr, pcl::PointCloud<PointT>::Ptr &cloudPtr, pcl::PolygonMesh::ConstPtr mesh, float resolution )
    {
        std::cout << "mesh2Octree...";

        cloudPtr = pcl::PointCloud<PointT>::Ptr( new pcl::PointCloud<PointT> );

        //Input of the above cloud and the corresponding output of cloud_pcl
        //pcl::fromPCLPointCloud2( mesh->cloud, *cloudPtr );
        pcl::fromROSMsg( mesh->cloud, *cloudPtr );

        octreePtr = OctreePtr( new Octree(resolution) );
        octreePtr->setInputCloud( cloudPtr );
        octreePtr->defineBoundingBox();
        octreePtr->addPointsFromInputCloud();

        //std::cout << "Octree Leaf Count Operation: " << octreePtr->getLeafCount() << std::endl;
        //std::cout << "Octree Branch Count Operation: " << octreePtr->getBranchCount() << std::endl;

        std::cout << "OK" << std::endl;
    }

    void
    MeshRayCaster::subdivideMesh( pcl::PolygonMesh &output_mesh, pcl::PolygonMesh::ConstPtr input_mesh, int iterations )
    {
#if 0
        //vtkPolyData *polyData = (vtkPolyData*)(dsActor->GetMapper()->GetInputAsDataSet());
        //vtkApproximatingSubdivisionFilter *filter = vtkCatmullClarkFilter::New();
        //filter->SetNumberOfSubdivisions( iterations );

        // Convert from PCL mesh representation to the VTK representation
        vtkSmartPointer<vtkPolyData> vtk_polygons;
        pcl::VTKUtils::convertToVTK( *input_mesh, vtk_polygons );

        // Apply the VTK algorithm
        vtkSmartPointer<vtkPolyDataAlgorithm> vtk_subdivision_filter = vtkCatmullClarkFilter::New();

        vtk_subdivision_filter->SetInput( vtk_polygons );
        vtk_subdivision_filter->Update();
        vtk_polygons = vtk_subdivision_filter->GetOutput();

        // Convert the result back to the PCL representation
        pcl::VTKUtils::convertToPCL( vtk_polygons, output_mesh );
#else
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
#endif

    }

    void
    MeshRayCaster::calculatePointDiffs( /* out: */
                                        /*  in: */
                                        OctreePtr const& octreePtr,
                                        pcl::PointCloud<PointT>::Ptr const& cloudPtr,
                                        Eigen::Affine3f const& pose,
                                        cv::Mat const& depth )
    {
        std::cout << "CalculatingDiffs...";
#if 0
        Eigen::Matrix3f rotation    = pose.rotation();
        Eigen::Vector3f translation = pose.translation();
        Eigen::Vector3f ray_next;

        for ( int y = 0; y < depthFC1.rows; ++y )
        {
            for ( int x = 0; x < depthFC1.cols; ++x)
            {
                ray_next = rotation * (get_next_ray( x, y, intrinsics_ ) * (float)depth.at<ushort>(y,x)) + translation;

                std::vector<int> pnt_indices;
                octreePtr->radiusSearch( const PointT &p_q, const double radius, std::vector<int> &k_indices,
                                         std::vector<float> &k_sqr_distances, unsigned int max_nn = 0) const;
                //( ray_start, ray_dir, pnt_indices, /* max_voxel_count: */ 1 );

                if ( pnt_indices.size() > 0 )
                {
                    p_indices.push_back( pnt_indices[0] );

                    PointT cloud_pnt3 = cloudPtr->at( pnt_indices[0] );
                    Eigen::Vector3f pnt3;
                    pnt3 << cloud_pnt3.x, cloud_pnt3.y, cloud_pnt3.z;
                    //depthFC1.at<float>(y,x) = static_cast<Eigen::Vector3f>(ray_start - pnt3).norm();
                }
            }
        }
#endif

    }

} // end ns am
