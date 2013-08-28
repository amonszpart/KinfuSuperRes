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

#ifndef MESH_H
#define	MESH_H

#include "GL/glew.h"

//#include <Importer.hpp>      // C++ importer interface
//#include <scene.h>       // Output data structure
//#include <postprocess.h> // Post processing flags

//#include "util.h"
#include "math_3d.h"
//#include "texture.h"
//#include "render_callbacks.h"
#include <Eigen/Dense>
#include <pcl/PolygonMesh.h>

#include <map>
#include <vector>

struct Vertex
{
        Vector3f m_pos;
        //Vector2f m_tex;
        Vector3f m_normal;
        //Vector3u m_normal;

        Vertex() {}

        Vertex( const Vector3f& pos/*, const Vector2f& tex*/, const Vector3f& normal )
        {
            m_pos    = pos;
            //m_tex    = tex;
            m_normal = normal;
        }
};

class Mesh
{
    public:
        Mesh();
        ~Mesh();

        bool loadMesh( std::string const& filename );
        bool loadMesh( pcl::PolygonMesh::Ptr const& meshPtr );
        void Render();

        GLuint vertexShaderLoc, normalShaderLoc;
        unsigned NumberOfMeshes() { return m_Entries.size(); }
    private:
        void initMesh( pcl::PolygonMesh::Ptr const& meshPtr );
        void clear();

#define INVALID_MATERIAL 0xFFFFFFFF

        struct MeshEntry
        {
                MeshEntry();
                ~MeshEntry();

                bool Init( const std::vector<Vertex>& Vertices,
                          const std::vector<unsigned int>& Indices );

                GLuint VB;
                GLuint IB;
                unsigned int NumIndices;
        };

        std::vector<MeshEntry> m_Entries;
        //std::vector<Texture*> m_Textures;

};


#endif	/* MESH_H */

