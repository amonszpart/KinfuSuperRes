#include "kinfu_util.h"

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////
vector<string>
getPcdFilesInDir(const string& directory)
{
    namespace fs = boost::filesystem;
    fs::path dir(directory);

    std::cout << "path: " << directory << std::endl;
    if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
        PCL_THROW_EXCEPTION (pcl::IOException, "No valid PCD directory given!\n");

    vector<string> result;
    fs::directory_iterator pos(dir);
    fs::directory_iterator end;

    for(; pos != end ; ++pos)
        if (fs::is_regular_file(pos->status()) )
            if (fs::extension(*pos) == ".pcd")
            {
#if BOOST_FILESYSTEM_VERSION == 3
                result.push_back (pos->path ().string ());
#else
                result.push_back (pos->path ());
#endif
                cout << "added: " << result.back() << endl;
            }

    return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
setViewerPose (visualization::PCLVisualizer& viewer, const Eigen::Affine3f& viewer_pose)
{
    Eigen::Vector3f pos_vector = viewer_pose * Eigen::Vector3f (0, 0, 0);
    Eigen::Vector3f look_at_vector = viewer_pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
    Eigen::Vector3f up_vector = viewer_pose.rotation () * Eigen::Vector3f (0, -1, 0);
    viewer.setCameraPosition (pos_vector[0], pos_vector[1], pos_vector[2],
            look_at_vector[0], look_at_vector[1], look_at_vector[2],
            up_vector[0], up_vector[1], up_vector[2]);
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

Eigen::Affine3f
getViewerPose (visualization::PCLVisualizer& viewer)
{
    Eigen::Affine3f pose = viewer.getViewerPose();
    Eigen::Matrix3f rotation = pose.linear();

    Matrix3f axis_reorder;
    axis_reorder << 0,  0,  1,
            -1,  0,  0,
            0, -1,  0;

    rotation = rotation * axis_reorder;
    pose.linear() = rotation;
    return pose;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

void
writePolygonMeshFile (int format, const pcl::PolygonMesh& mesh, std::string fileName)
{
    if (format == nsKinFuApp::MESH_PLY)
    {
        cout << "Saving mesh to '" + fileName + "_mesh.ply'... " << flush;
        pcl::io::savePLYFile( fileName + "_mesh.ply", mesh );
    }
    else /* if (format == KinFuApp::MESH_VTK) */
    {
        cout << "Saving mesh to '" + fileName + "_mesh.vtk'... " << flush;
        pcl::io::saveVTKFile(fileName + "_mesh.vtk", mesh);
    }
    cout << "Done" << endl;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{
    if (triangles.empty())
    {
        std::cerr << "kinfu_util::convertToMesh(): triangles empty...returning null..." << std::endl;
        return boost::shared_ptr<pcl::PolygonMesh>();
    }

    pcl::PointCloud<pcl::PointXYZ> cloud;
    cloud.width  = (int)triangles.size();
    cloud.height = 1;
    triangles.download(cloud.points);

    boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() );
    pcl::toROSMsg(cloud, mesh_ptr->cloud);

    mesh_ptr->polygons.resize (triangles.size() / 3);
    for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
    {
        pcl::Vertices v;
        v.vertices.push_back(i*3+0);
        v.vertices.push_back(i*3+2);
        v.vertices.push_back(i*3+1);
        mesh_ptr->polygons[i] = v;
    }
    return mesh_ptr;
}

