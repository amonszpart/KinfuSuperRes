#include "scene_cloud_view.h"

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{
    if (triangles.empty())
        return boost::shared_ptr<pcl::PolygonMesh>();

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
