#ifndef TRIANGLERENDERER_H
#define TRIANGLERENDERER_H

#include <GL/glew.h>
#include <GL/freeglut.h>

#include "mesh.h"

#include <eigen3/Eigen/Dense>
#include <opencv2/core/core.hpp>


namespace am
{
    class TriangleRenderer
    {
        public:
            /*
             *\brief      Essential method, renders a mesh, and returns it's depthmap and vertex- and triangle-index matrices.
             *\param[OUT] depths    vector<Mat> of size 2, depths[0] contains distances from camera in 0.f..alpha range
             *                                             depths[1] contains unusable (clamped) Ids //FIXME scale to 0..1.f in ".frag"
             *\param[OUT] indices   vector<Mat> of size 2, indices[0] contains vertexId as unsigned,
             *                                             indices[1] contains triangleId as unsigned
             *\param[IN]  w         image width
             *\param[IN]  h         image height
             *\param[IN]  intrinsics Intrinsics matrix 3x3
             *\param[IN]  pose      camera position
             *\param[IN]  meshPtr   input mesh pointer to render
             *\param[IN]  alpha     scales the depths[0] output
             */
            void renderDepthAndIndices( std::vector<cv::Mat> &depth, std::vector<cv::Mat> &indices,
                                       int w, int h, Eigen::Matrix3f const& intrinsics,
                                       Eigen::Affine3f const& pose, pcl::PolygonMesh::Ptr const& meshPtr, float alpha = 10001.f );
            /*
             *\brief Loads mesh from path, and calls renderDepthAndIndices with the filled meshPtr
             */
            void renderDepthAndIndices( std::vector<cv::Mat> &depth, std::vector<cv::Mat> &indices,
                                        int w, int h, Eigen::Matrix3f const& intrinsics,
                                        Eigen::Affine3f const& pose, std::string const& meshPath, float alpha = 10001.f );

            /*
             *\brief    Read framebuffer object's distance data
             *\param[OUT] distances contains distances from camera in 0.f..alpha range
             *\param[IN]  alpha     scales distances
             *\param[OUT] indices   contains unusable (clamped) Ids // FIXME scale to 0..1.f by vertexcount in ".frag"
             */
            void readDepthToFC1( cv::Mat &distances, float alpha, cv::Mat &indices );
            /*
             *\brief    Read framebuffer object's vertex and triangle index data
             *\param[OUT] vertexIds   contains vertex indices as unsigned integers stored in a CV_8UC4 format, read with ".at<unsigned>(y,x)"
             *\param[OUT] triangleIDs contains triangle indices as unsigned integers stored in a CV_8UC4 format, read with ".at<unsigned>(y,x)"
             */
            void readIds( cv::Mat &vertexIds, cv::Mat &triangleIds );


            TriangleRenderer();
            ~TriangleRenderer();

            int         W       () { return width_;  }
            int         H       () { return height_; }
            GLubyte *&  Pixels  () { return pixels;  }
            GLuint  *&  Ids     () { return ids;     }
            int         SzPixels() { return width_ * height_ * sizeof(float   ) * 4; }
            int         SzIds   () { return width_ * height_ * sizeof(unsigned) * 3; }

            /// METHODS
            // shaders
            GLuint setupShaders();
            void setupBuffers( int width, int height );
            void setSize( int w, int h );
            void init(int w, int h);
            void loadMesh( std::string const& mesh_path );

            // projection matrix
            void setIntrinsics(Eigen::Matrix3f K, float zNear, float zFar );

            // render
            void setCamera( float posX, float posY, float posZ,
                            float lookAtX, float lookAtY, float lookAtZ,
                            float upX, float upY, float upZ );
            void setCamera( Eigen::Affine3f const& pose );
            void setUniforms();
            void renderScene( void );


        protected:
            GLuint depthRenderBufferHandle_;
            GLuint framebufferHandle_;
            GLuint textureHandles_[2];

            Mesh meshes_;

            GLuint shader_program, vertex_shader, fragment_shader;

            // Uniform variable Locations
            GLuint projMatrixLoc, viewMatrixLoc, modelMatrixLoc, eyeLoc;
            // Vertex Attribute Locations
            GLuint vertexLoc, normalLoc;

            // storage for Matrices
            float projMatrix [16];
            float viewMatrix_ [16];
            float modelMatrix_[16];
            float eyePosition_[3 ];
            Eigen::Matrix3f intrinsics_;

            int      width_;
            int      height_;
            GLubyte *pixels;
            GLuint  *ids;
            int      szPixels;
            bool     inited_;

            // Shader Names
            const char *vertexFileName   = "../TriangleTracing/build/shaders/triangles.vert"; // copied into build directory by cmake
            const char *fragmentFileName = "../TriangleTracing/build/shaders/triangles.frag";
    };

} // end ns am

#endif // TRIANGLERENDERER_H
