/*

	Copyright 2011 Etay Meiri

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <assert.h>

#include "mesh.h"

#include <pcl/io/ply_io.h>
#include <pcl/io/vtk_lib_io.h>

#define INVALID_OGL_VALUE (0xFFFFFFFF)

Mesh::MeshEntry::MeshEntry()
{
    VB = INVALID_OGL_VALUE;
    IB = INVALID_OGL_VALUE;
    NumIndices  = 0;
};

Mesh::MeshEntry::~MeshEntry()
{
    if (VB != INVALID_OGL_VALUE)
    {
        glDeleteBuffers(1, &VB);
    }

    if (IB != INVALID_OGL_VALUE)
    {
        glDeleteBuffers(1, &IB);
    }
}

bool Mesh::MeshEntry::Init(const std::vector<Vertex>& Vertices,
                          const std::vector<unsigned int>& Indices)
{
    NumIndices = Indices.size();

    glGenBuffers( 1, &VB );
  	glBindBuffer(GL_ARRAY_BUFFER, VB);
	glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * Vertices.size(), &Vertices[0], GL_STATIC_DRAW);

    glGenBuffers( 1, &IB );
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, IB);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(unsigned int) * NumIndices, &Indices[0], GL_STATIC_DRAW);
    
    return glGetError() == GL_NO_ERROR;
}

Mesh::Mesh()
{

}


Mesh::~Mesh()
{
    clear();
}


void Mesh::clear()
{
    m_Entries.clear();
}


bool Mesh::loadMesh( const std::string& filename )
{
    pcl::PolygonMesh::Ptr meshPtr( new pcl::PolygonMesh );
    pcl::io::loadPolygonFile( filename, *meshPtr );

    return loadMesh( meshPtr );
}

bool Mesh::loadMesh( pcl::PolygonMesh::Ptr const& meshPtr )
{
    // Release the previously loaded mesh (if it exists)
    clear();

    initMesh( meshPtr );

    return true;
}

void Mesh::initMesh( pcl::PolygonMesh::Ptr const& meshPtr )
{
    const int MAX_VERTICES = 100000000;

    std::vector<Vertex> vertices;
    std::vector<unsigned int> indices;
#if 1
    const int point_step = meshPtr->cloud.point_step;
    const int x_offs     = meshPtr->cloud.fields[0].offset;
    const int y_offs     = meshPtr->cloud.fields[1].offset;
    const int z_offs     = meshPtr->cloud.fields[2].offset;
    const int rgb_offs   = meshPtr->cloud.fields[3].offset;

    float maxx, maxy, maxz;
    float minx, miny, minz;

    std::cout << "Mesh::InitMesh(): reading " << meshPtr->cloud.data.size() / point_step << " vertices..." << std::endl;
    for ( int pid = 0; pid < meshPtr->cloud.data.size(); pid+= point_step )
    {
        float *p_x = reinterpret_cast<float*>( &(meshPtr->cloud.data[pid + x_offs]) );
        float *p_y = reinterpret_cast<float*>( &(meshPtr->cloud.data[pid + y_offs]) );
        float *p_z = reinterpret_cast<float*>( &(meshPtr->cloud.data[pid + z_offs]) );
        //unsigned char *p_bgr = reinterpret_cast<uchar*>( &(meshPtr->cloud.data[pid + rgb_offs]) );

        maxx = maxy = maxz = 0.f;
        minx = miny = minz = FLT_MAX;
        if ( *p_x > maxx ) maxx = *p_x ;
        if ( *p_y > maxy ) maxy = *p_y ;
        if ( *p_z > maxz ) maxz = *p_z ;
        if ( *p_x < minx ) minx = *p_x ;
        if ( *p_y < miny ) miny = *p_y ;
        if ( *p_z < minz ) minz = *p_z ;

        int fid = pid/point_step/3;

        vertices.push_back(
                    Vertex(Vector3f(*p_x, *p_y, *p_z),Vector3f( /*  vertexId: */ pid/point_step,
                                                                /* polygonId: */ fid+1,
                                                                /*    unused: */ 1.f))
                    );

        if ( fid == 71935 )
        {
            std::cout << "pid: " << pid
                      << " vxid: " << pid / point_step
                      << " fid: " << fid
                      << " vertices: "
                      << meshPtr->polygons[fid].vertices[0] << ","
                      << meshPtr->polygons[fid].vertices[1] << ","
                      << meshPtr->polygons[fid].vertices[2] << std::endl;
        }
        //std::cout << pid/point_step << ": "
        //          << meshPtr->polygons[pid/point_step/3].vertices[0] << "," << meshPtr->polygons[pid/point_step/3].vertices[1] << "," << meshPtr->polygons[pid/point_step/3].vertices[2] << std::endl;

        /*std::cout << "putting vertex: "
                  << vertices.back().m_pos.x << ","
                  << vertices.back().m_pos.y << ","
                  << vertices.back().m_pos.z << std::endl;*/
        if ( vertices.size() >= MAX_VERTICES ) break;
    }

    std::cout << "Mesh::InitMesh(): reading " << meshPtr->polygons.size() << " faces..." << std::endl;
    unsigned maxFaceId = 0;
    for ( int fid = 0; fid < meshPtr->polygons.size(); ++fid )
    {
        assert( meshPtr->polygons[fid].vertices.size() == 3 );

        if ( (meshPtr->polygons[fid].vertices[0] < MAX_VERTICES) &&
             (meshPtr->polygons[fid].vertices[1] < MAX_VERTICES) &&
             (meshPtr->polygons[fid].vertices[2] < MAX_VERTICES) )
        {
        indices.push_back( meshPtr->polygons[fid].vertices[2] );
        if ( indices.back() > maxFaceId ) maxFaceId = indices.back();
        indices.push_back( meshPtr->polygons[fid].vertices[1] );
        if ( indices.back() > maxFaceId ) maxFaceId = indices.back();
        indices.push_back( meshPtr->polygons[fid].vertices[0] );
        if ( indices.back() > maxFaceId ) maxFaceId = indices.back();
        }
    }
    std::cout << "maxfaceID: " << maxFaceId << std::endl;
    std::cout << "verticescount: " << vertices.size() << std::endl;

    m_Entries.push_back( MeshEntry() );
    m_Entries.back().Init( vertices, indices );
#endif
#if 0
    // test
    float fid = 0.f;
    m_Entries.push_back( MeshEntry() );
    vertices.clear();


    // back
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 3.f), Vector3f(fid++, 0.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 3.f, 3.f), Vector3f(fid++, 0.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 3.f), Vector3f(fid++, 0.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 3.f), Vector3f(fid++, 0.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 3.f), Vector3f(fid++, 0.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 3.f), Vector3f(fid++, 0.f, 1.f)) );

    // left
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 0.f), Vector3f(fid++, 0.3f, 0.6f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 3.f), Vector3f(fid++, 0.3f, 0.6f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 3.f, 3.f), Vector3f(fid++, 0.3f, 0.6f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 3.f, 3.f), Vector3f(fid++, 0.3f, 0.6f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 3.f, 0.f), Vector3f(fid++, 0.3f, 0.6f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 0.f), Vector3f(fid++, 0.3f, 0.6f)) );
    // right
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 0.f), Vector3f(fid++, 0.6f, 0.3f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 3.f), Vector3f(fid++, 0.6f, 0.3f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 3.f), Vector3f(fid++, 0.6f, 0.3f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 3.f), Vector3f(fid++, 0.6f, 0.3f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 0.f), Vector3f(fid++, 0.6f, 0.3f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 0.f), Vector3f(fid++, 0.6f, 0.3f)) );

    // bottom
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 0.f), Vector3f(fid++, 1.f, 0.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 0.f), Vector3f(fid++, 1.f, 0.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 3.f), Vector3f(fid++, 1.f, 0.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 3.f), Vector3f(fid++, 1.f, 0.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 3.f), Vector3f(fid++, 1.f, 0.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 0.f), Vector3f(fid++, 1.f, 0.f)) );

#if 0
    // top
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 3.f), Vector3f(fid++, 0.3f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 3.f, 3.f), Vector3f(fid++, 0.3f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 3.f), Vector3f(fid++, 0.3f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 3.f), Vector3f(fid++, 0.3f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 3.f), Vector3f(fid++, 0.3f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 3.f), Vector3f(fid++, 0.3f, 1.f)) );
#endif
#if 0
    // front
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 0.f), Vector3f(fid++, 1.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 3.f, 0.f), Vector3f(fid++, 1.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 0.f), Vector3f(fid++, 1.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 3.f, 0.f), Vector3f(fid++, 1.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(3.f, 0.f, 0.f), Vector3f(fid++, 1.f, 1.f)) );
    vertices.push_back( Vertex(Vector3f(0.f, 0.f, 0.f), Vector3f(fid++, 1.f, 1.f)) );
#endif


    indices.clear();
    for ( int i = 0; i < vertices.size(); ++i )
        indices.push_back( i );
    m_Entries.back().Init( vertices, indices );
#endif
}

void Mesh::Render()
{
    glEnableVertexAttribArray( vertexShaderLoc );
    glEnableVertexAttribArray( normalShaderLoc );
    
    for (unsigned int i = 0 ; i < m_Entries.size() ; i++)
    {
        std::cout << "rendering " << i << std::endl;

        glBindBuffer( GL_ARRAY_BUFFER, m_Entries[i].VB );
        glVertexAttribPointer( vertexShaderLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
        glVertexAttribPointer( normalShaderLoc, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);
        //glVertexAttribPointer( normalShaderLoc, 3, GL_UNSIGNED_INT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);

        glBindBuffer( GL_ELEMENT_ARRAY_BUFFER, m_Entries[i].IB );

        glDrawElements( GL_TRIANGLES, m_Entries[i].NumIndices, GL_UNSIGNED_INT, 0 );
    }

    glDisableVertexAttribArray( vertexShaderLoc );
    glDisableVertexAttribArray( normalShaderLoc );
}

#if 0
void Mesh::Render(unsigned int DrawIndex, unsigned int PrimID)
{
    assert(DrawIndex < m_Entries.size());
    
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);

    glBindBuffer(GL_ARRAY_BUFFER, m_Entries[DrawIndex].VB);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), 0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)12);
    glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (const GLvoid*)20);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, m_Entries[DrawIndex].IB);

    glDrawElements(GL_TRIANGLES, 3, GL_UNSIGNED_INT, (const GLvoid*)(PrimID * 3 * sizeof(GLuint)));

    glDisableVertexAttribArray(0);
    glDisableVertexAttribArray(1);
    glDisableVertexAttribArray(2);    
}
#endif
