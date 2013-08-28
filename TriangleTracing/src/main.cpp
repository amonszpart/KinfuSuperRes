#include <iostream>

using namespace std;

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "textfile.h"

#define M_PI       3.14159265358979323846

#include "mesh.h"

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <pcl/io/vtk_lib_io.h>
#include <eigen3/Eigen/Dense>

#define INVALID_OGL_VALUE (0xFFFFFFFF)

struct GLRenderer
{
        void renderDepthAndIndices( std::vector<cv::Mat> &depth, std::vector<cv::Mat> &indices,
                                 int w, int h, Eigen::Matrix3f const& intrinsics,
                                 Eigen::Affine3f const& pose, std::string const& meshPath, float alpha = 10001.f );

        void renderDepthAndIndices(std::vector<cv::Mat> &depth, std::vector<cv::Mat> &indices,
                                 int w, int h, Eigen::Matrix3f const& intrinsics,
                                 Eigen::Affine3f const& pose, pcl::PolygonMesh::Ptr const& meshPtr, float alpha = 10001.f );

        GLRenderer();
        ~GLRenderer();

        int         W       () { return width_;  }
        int         H       () { return height_; }
        GLubyte *&  Pixels  () { return pixels;  }
        GLuint  *&  Ids     () { return ids;     }
        int         SzPixels() { return width_ * height_ * sizeof(float   ) * 4; }
        int         SzIds   () { return width_ * height_ * sizeof(unsigned) * 3; }

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

        // ReadPixels
        void readDepthToFC1( cv::Mat &mat, float alpha, cv::Mat &indices );
        void readIds       ( cv::Mat &mat, cv::Mat &triangleIDs );

    protected:
        int      width_;
        int      height_;
        GLubyte *pixels;
        GLuint  *ids;
        int      szPixels;
        bool    inited_;
} rendererInstance;

void GLRenderer::setSize( int w, int h )
{
    width_  = w;
    height_ = h;
    if ( pixels ) { delete [] pixels; pixels = NULL; }
    if ( ids    ) { delete [] ids   ; ids    = NULL; }

    pixels = new GLubyte[ width_*height_*sizeof(float   ) * 4 ];
    ids    = new GLuint [ width_*height_*sizeof(unsigned) * 3 ];

    this->setupBuffers( width_, height_ );
}

GLRenderer::GLRenderer()
    : width_(1280), height_(960),
      pixels( new GLubyte[width_*height_*sizeof(float   ) * 4] ),
      ids   ( new GLuint [width_*height_*sizeof(unsigned) * 3] ),
      depthRenderBufferHandle_( INVALID_OGL_VALUE ), framebufferHandle_( INVALID_OGL_VALUE ),
      inited_(false)
{
    textureHandles_[0] = textureHandles_[1] = INVALID_OGL_VALUE;
}

GLRenderer::~GLRenderer()
{
    if ( pixels ) { delete [] pixels; pixels = NULL; }
    if ( ids    ) { delete [] ids   ; ids    = NULL; }

    glDeleteTextures     ( 2, textureHandles_    );
    glDeleteFramebuffers ( 1, &framebufferHandle_   );
    glDeleteRenderbuffers( 1, &depthRenderBufferHandle_ );

    glDeleteProgram( shader_program );
    glDeleteShader ( vertex_shader );
    glDeleteShader ( fragment_shader );
}

void GLRenderer::init( int w, int h )
{
    std::cout << "GLRenderer initing..." << std::endl;

    char *myargv[1]; myargv[0] = strdup( "GLRendererGlutApp" );
    int   myargc = 1;
    glutInit( &myargc, myargv );

    glutInitDisplayMode( GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA );
    ///glutInitWindowPosition(100,100);
    //glutInitWindowSize( gInst.W(), gInst.H() );
    int windowHandle = glutCreateWindow("GLRendererGlutApp");

    ///glutDisplayFunc(renderScene);
    ///glutIdleFunc(renderScene);
    //glutReshapeFunc(changeSize);
    //glutKeyboardFunc(processNormalKeys);

    glewInit();
    if ( glewIsSupported("GL_VERSION_3_3") ) printf( "Ready for OpenGL 3.3\n" );
    else {
        printf( "OpenGL 3.3 not supported\n" );
        exit(1);
    }

    glEnable( GL_DEPTH_TEST );
    glClearColor( 0.0, 0.0, 0.0, 1.0 );

    shader_program             = setupShaders();
    meshes_.vertexShaderLoc    = vertexLoc;
    meshes_.normalShaderLoc    = normalLoc;
    inited_ = true;

    this->setSize( w, h );
}

// "/home/bontius/workspace/rec/testing/cloud_mesh.ply"
void GLRenderer::loadMesh( std::string const& mesh_path )
{
    meshes_.loadMesh( mesh_path );
    meshes_.vertexShaderLoc    = vertexLoc;
    meshes_.normalShaderLoc    = normalLoc;
}

// Shader Names
const char *vertexFileName   = "shaders/triangles.vert";
const char *fragmentFileName = "shaders/triangles.frag";

// ----------------------------------------------------
// VECTOR STUFF
//

// res = a cross b;
void crossProduct( float *a, float *b, float *res )
{
    res[0] = a[1] * b[2]  -  b[1] * a[2];
    res[1] = a[2] * b[0]  -  b[2] * a[0];
    res[2] = a[0] * b[1]  -  b[0] * a[1];
}

// Normalize a vec3
void normalize( float *a )
{
    float mag = sqrt(a[0] * a[0]  +  a[1] * a[1]  +  a[2] * a[2]);

    a[0] /= mag;
    a[1] /= mag;
    a[2] /= mag;
}

// ----------------------------------------------------
// MATRIX STUFF
//

// sets the square matrix mat to the identity matrix,
// size refers to the number of rows (or columns)
void setIdentityMatrix( float *mat, int size )
{
    // fill matrix with 0s
    for (int i = 0; i < size * size; ++i)
        mat[i] = 0.0f;

    // fill diagonal with 1s
    for (int i = 0; i < size; ++i)
        mat[i + i * size] = 1.0f;
}

//
// a = a * b;
//
void multMatrix( float *a, float *b )
{
    float res[16];

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            res[j*4 + i] = 0.0f;
            for (int k = 0; k < 4; ++k) {
                res[j*4 + i] += a[k*4 + i] * b[j*4 + k];
            }
        }
    }
    memcpy(a, res, 16 * sizeof(float));
}

// Defines a transformation matrix mat with a translation
void setTranslationMatrix( float *mat, float x, float y, float z )
{
    setIdentityMatrix( mat, 4 );
    mat[12] = x;
    mat[13] = y;
    mat[14] = z;
}

// ----------------------------------------------------
// Projection Matrix
//

void buildProjectionMatrix( float* projMatrix, float fov, float ratio, float nearP, float farP )
{
    float f = 1.0f / tan (fov * (M_PI / 360.0));

    setIdentityMatrix( projMatrix, 4 );

    projMatrix[0] = f / ratio;
    projMatrix[1 * 4 + 1] = f;
    projMatrix[2 * 4 + 2] = (farP + nearP) / (nearP - farP);
    projMatrix[3 * 4 + 2] = (2.0f * farP * nearP) / (nearP - farP);
    projMatrix[2 * 4 + 3] = -1.0f;
    projMatrix[3 * 4 + 3] = 0.0f;
}

// ----------------------------------------------------
// View Matrix
//
// note: it assumes the camera is not tilted,
// i.e. a vertical up vector (remmeber gluLookAt?)
//

void GLRenderer::setCamera( float posX, float posY, float posZ,
                            float lookAtX, float lookAtY, float lookAtZ,
                            float upX, float upY, float upZ )
{

    float dir[3], right[3], up[3];

    up[0] = upX;   up[1] = upY;   up[2] = upZ;

    dir[0] =  (lookAtX - posX);
    dir[1] =  (lookAtY - posY);
    dir[2] =  (lookAtZ - posZ);
    normalize(dir);

    crossProduct( dir, up, right );
    normalize(right);

    crossProduct(right,dir,up);
    normalize(up);

    float aux[16];

    viewMatrix_[0]  = right[0];
    viewMatrix_[4]  = right[1];
    viewMatrix_[8]  = right[2];
    viewMatrix_[12] = 0.0f;

    viewMatrix_[1]  = -up[0];
    viewMatrix_[5]  = -up[1];
    viewMatrix_[9]  = -up[2];
    viewMatrix_[13] = 0.0f;

    viewMatrix_[2]  = -dir[0];
    viewMatrix_[6]  = -dir[1];
    viewMatrix_[10] = -dir[2];
    viewMatrix_[14] =  0.0f;

    viewMatrix_[3]  = 0.0f;
    viewMatrix_[7]  = 0.0f;
    viewMatrix_[11] = 0.0f;
    viewMatrix_[15] = 1.0f;

    setTranslationMatrix( aux, -posX, -posY, -posZ );

    multMatrix( viewMatrix_, aux );

    eyePosition_[0] = posX;
    eyePosition_[1] = posY;
    eyePosition_[2] = posZ;

    setIdentityMatrix( modelMatrix_, 4 );
}

// ----------------------------------------------------

void changeSize( int w, int h )
{

    float ratio;
    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if ( h == 0 ) h = 1;

    // Set the viewport to be the entire window
    glViewport( 0, 0, w, h );

    //ratio = (1.0f * w) / h;
    //buildProjectionMatrix(53.13f, ratio, 1.0f, 30.0f);
    std::cerr << "changeSize not implemented..." << std::endl;
}

void GLRenderer::setupBuffers( int width, int height )
{
    if ( !inited_ ) { std::cerr << "GLRenderer::setupBuffers(): cannot run, buffers not inited!" << std::endl; return; }

    if (       textureHandles_[0] != INVALID_OGL_VALUE ) glDeleteTextures    ( 2,  textureHandles_          );
    if ( depthRenderBufferHandle_ != INVALID_OGL_VALUE ) glDeleteTextures    ( 1, &depthRenderBufferHandle_ );
    if (       framebufferHandle_ != INVALID_OGL_VALUE ) glDeleteFramebuffers( 1, &framebufferHandle_       );

    glGenFramebuffers( 1, &framebufferHandle_ );
    glBindFramebuffer(GL_FRAMEBUFFER, framebufferHandle_ );
    {
        glGenTextures( 2, textureHandles_ );
        glBindTexture( GL_TEXTURE_2D, textureHandles_[0] );

        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureHandles_[0], 0 );
    }

    // Create the texture object for the primitive information buffer
    {
        glBindTexture(GL_TEXTURE_2D, textureHandles_[1] );
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32UI, width, height, 0, GL_RGB_INTEGER, GL_UNSIGNED_INT, NULL );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, textureHandles_[1], 0 );
        //glReadPixels(x, y, 1, 1, GL_RGB_INTEGER, GL_UNSIGNED_INT, &Pixel);
    }

    // The depth buffer
    {
        glGenRenderbuffers(1, &depthRenderBufferHandle_);
        glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBufferHandle_);
        glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );
        glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBufferHandle_ );
    }

    // Set the list of draw buffers.
    GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers( 2, DrawBuffers ); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "FrameBufferStatus error..." << std::endl;
}

void GLRenderer::setUniforms()
{
    // must be called after glUseProgram
    glUniformMatrix4fv(projMatrixLoc,  1, false,  projMatrix );
    glUniformMatrix4fv(viewMatrixLoc,  1, false,  viewMatrix_ );
    glUniformMatrix4fv(modelMatrixLoc, 1, false, modelMatrix_ );
    glUniform3f( eyeLoc, eyePosition_[0], eyePosition_[1], eyePosition_[2] );
}

void GLRenderer::setCamera( Eigen::Affine3f const& pose )
{
    Eigen::Vector3f pos_vector     = pose * Eigen::Vector3f (0, 0, 0);
    Eigen::Vector3f look_at_vector = pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
    Eigen::Vector3f up_vector      = pose.rotation () * Eigen::Vector3f (0, -1, 0);

    setCamera( pos_vector[0], pos_vector[1], pos_vector[2],
               look_at_vector[0], look_at_vector[1], look_at_vector[2],
               up_vector[0], up_vector[1], up_vector[2] );
}

void GLRenderer::renderScene()
{
    if ( !inited_                  ) { std::cerr << "GLRenderer::renderScene(): cannot run, buffers not inited!" << std::endl; return; }
    if ( !meshes_.NumberOfMeshes() ) { std::cerr << "GLRenderer::renderScene(): cannot run, no meshes!"          << std::endl; return; }

    glUseProgram( shader_program );
    setUniforms();

    // set render target
    glBindFramebuffer( GL_FRAMEBUFFER, framebufferHandle_ );
    glDrawBuffers( 2, textureHandles_ );
    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    glViewport( 0, 0, width_, height_ ); // Render on the whole framebuffer, complete from the lower left corner to the upper right

    // render
    meshes_.Render();

    // reset render target
    GLenum tmpBuff[] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers( 1, tmpBuff );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    // render again
    //glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
    //meshes_.Render();

    //glutSwapBuffers();
}

void processNormalKeys( unsigned char key, int x, int y )
{
    if ( key == 27 )
    {
        //glDeleteVertexArrays(3,vao);

        exit(0);
    }
}

#define printOpenGLError() printOglError(__FILE__, __LINE__)

int printOglError(char *file, int line)
{
    //
    // Returns 1 if an OpenGL error occurred, 0 otherwise.
    //
    GLenum glErr;
    int    retCode = 0;

    glErr = glGetError();
    while (glErr != GL_NO_ERROR)
    {
        printf("glError in file %s @ line %d: %s\n", file, line, gluErrorString(glErr));
        retCode = 1;
        glErr = glGetError();
    }
    return retCode;
}

void printShaderInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;

    glGetShaderiv(obj, GL_INFO_LOG_LENGTH,&infologLength);

    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetShaderInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s\n",infoLog);
        free(infoLog);
    }
}

void printProgramInfoLog(GLuint obj)
{
    int infologLength = 0;
    int charsWritten  = 0;
    char *infoLog;

    glGetProgramiv(obj, GL_INFO_LOG_LENGTH,&infologLength);

    if (infologLength > 0)
    {
        infoLog = (char *)malloc(infologLength);
        glGetProgramInfoLog(obj, infologLength, &charsWritten, infoLog);
        printf("%s\n",infoLog);
        free(infoLog);
    }
}

GLuint GLRenderer::setupShaders()
{
    char *vs = NULL, *fs = NULL, *fs2 = NULL;

    vertex_shader   = glCreateShader( GL_VERTEX_SHADER   );
    fragment_shader = glCreateShader( GL_FRAGMENT_SHADER );

    vs = textFileRead( vertexFileName   );
    fs = textFileRead( fragmentFileName );

    const char * vv = vs;
    const char * ff = fs;

    glShaderSource( vertex_shader, 1, &vv, NULL );
    glShaderSource( fragment_shader, 1, &ff, NULL );

    free(vs); free(fs);

    glCompileShader( vertex_shader   );
    glCompileShader( fragment_shader );

    printShaderInfoLog( vertex_shader   );
    printShaderInfoLog( fragment_shader );

    shader_program = glCreateProgram();
    glAttachShader( shader_program, vertex_shader   );
    glAttachShader( shader_program, fragment_shader );

    glBindFragDataLocation( shader_program, 0, "outputF" );
    glBindFragDataLocation( shader_program, 1, "ids"     );
    glLinkProgram( shader_program );
    printProgramInfoLog( shader_program );

    vertexLoc = glGetAttribLocation(  shader_program, "position" );
    //texLoc    = glGetAttribLocation(  shader_program, "texCoord" );
    normalLoc  = glGetAttribLocation( shader_program, "normal"   );

    projMatrixLoc  = glGetUniformLocation( shader_program, "projMatrix"  );
    viewMatrixLoc  = glGetUniformLocation( shader_program, "viewMatrix"  );
    modelMatrixLoc = glGetUniformLocation( shader_program, "modelMatrix" );
    eyeLoc         = glGetUniformLocation( shader_program, "eyePos"      );

    return( shader_program );
}

void GLRenderer::setIntrinsics( Eigen::Matrix3f K, float zNear, float zFar )
{
    const float x0 = 0.f;
    const float y0 = 0.f;
    const float w = (float) width_;
    const float h = (float)height_;

    intrinsics_ = K;
    float proj[16] = {  2.f * K(0,0) / w, -2.f * K(0,1) / w,                  0.f,                                   0.f,
                        0.f             ,  2.f * K(1,1) / h,                  0.f,                                   0.f,
                        0.f             , ( w - 2.f * K(0,2) + 2.f * x0) / w, (-zFar - zNear) / (zFar - zNear)    , -1.f,
                        0.f             , (-h + 2.f * K(1,2) + 2.f * y0) / h, -2.f * zFar * zNear / (zFar - zNear),  0.f };

    memcpy( projMatrix, proj, 16 * sizeof(float) );
}

void GLRenderer::readDepthToFC1( cv::Mat &mat, float alpha, cv::Mat &indices )
{
    const int pixels_point_step = sizeof(float) * 4;

    glBindFramebuffer( GL_FRAMEBUFFER, framebufferHandle_ );
    glReadBuffer( GL_COLOR_ATTACHMENT0 );
    glReadPixels( 0, 0, width_, height_, GL_RGBA, GL_FLOAT, pixels );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    mat.create( height_, width_, CV_32FC1 );
    indices.create( height_, width_, CV_32FC1 );
    for ( int y = 0; y < height_; ++y )
    {
        for ( int x = 0; x < width_; ++x )
        {
            // copy one channel only
            float val = *reinterpret_cast<float*>( &pixels[(y * width_ + x) * pixels_point_step] ) * alpha;
            mat.at<float>( y, x ) = val;
            float ind = *reinterpret_cast<float*>( &pixels[(y * width_ + x) * pixels_point_step + sizeof(float)] );
            indices.at<float>( y, x) = ind;
        }
    }
}

void GLRenderer::readIds( cv::Mat &inds, cv::Mat &triangleIDs )
{
    glBindFramebuffer( GL_FRAMEBUFFER, framebufferHandle_ );
    glReadBuffer( GL_COLOR_ATTACHMENT1 );
    glReadPixels( 0, 0, width_, height_, GL_RGB_INTEGER, GL_UNSIGNED_INT, ids );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    inds.create( height_, width_, CV_8UC4 );
    triangleIDs.create( height_, width_, CV_8UC4 );
    for ( int y = 0; y < height_; ++y )
    {
        for ( int x = 0; x < width_; ++x )
        {
            // copy one channel only

            inds       .at<unsigned>(y,x) = ids[ (y * width_ + x) * 3     ]; // ids is GLuint pointer, so sizeof unsigned not needed
            triangleIDs.at<unsigned>(y,x) = ids[ (y * width_ + x) * 3 + 1 ];
        }
    }
}


void GLRenderer::renderDepthAndIndices(std::vector<cv::Mat> &depths, std::vector<cv::Mat> &indices,
                                     int w, int h, Eigen::Matrix3f const& intrinsics,
                                     Eigen::Affine3f const& pose, std::string const& meshPath,
                                     float alpha )
{
    pcl::PolygonMesh::Ptr meshPtr( new pcl::PolygonMesh );
    pcl::io::loadPolygonFile( meshPath, *meshPtr );

    renderDepthAndIndices( depths, indices, w, h, intrinsics, pose, meshPtr, alpha );
}

void GLRenderer::renderDepthAndIndices( std::vector<cv::Mat> &depths, std::vector<cv::Mat> &indices,
                                     int w, int h, Eigen::Matrix3f const& intrinsics,
                                     Eigen::Affine3f const& pose, pcl::PolygonMesh::Ptr const& meshPtr,
                                     float alpha )
{
    if ( !inited_ ) init( w, h );

    setIntrinsics( intrinsics, 0.001, 10.01 );
    setCamera    ( pose );

    // model
    meshes_.loadMesh( meshPtr );

    //glutMainLoopEvent();
    //glutMainLoop();
    renderScene();

    depths.resize(2);
    indices.resize(2);
    readDepthToFC1( depths[0], alpha, depths[1] );
    readIds( indices[0], indices[1] );
}

int main( int argc, char **argv )
{
    // projection
    Eigen::Matrix3f intrinsics;
//    intrinsics << 521.7401 * 2.f, 0       , 323.4402 * 2.f,
//            0             , 522.1379 * 2.f, 258.1387 * 2.f,
//            0             , 0             , 1             ;
    intrinsics << 521.7401 * 2.f / 1280.f * 32.f, 0.f       , 323.4402 * 2.f / 1280.f * 32.f,
            0.f             , 522.1379 * 2.f / 960.f * 24.f, 258.1387 * 2.f / 960.f * 24.f,
            0.f             , 0.f             , 1.f             ;

    // view
    Eigen::Affine3f pose; pose.linear()
            << 1.f, 0.f, 0.f,
               0.f, 1.f, 0.f,
               0.f, 0.f, 1.f;
    pose.translation() << 1.5f, 1.5f, -0.3f;

    std::vector<cv::Mat> depths, indices;
    cv::Mat depthFC1, indices32UC1;
    rendererInstance.renderDepthAndIndices( depths, indices, 32, 24, intrinsics, pose, "/home/bontius/workspace/rec/testing/cloud_mesh.ply", 10001.f );

    depthFC1     = depths[0];  // depths
    indices32UC1 = indices[0]; // vertex ids

    cv::imshow( "depth", depthFC1 / 5001.f );
    for ( int y = 0; y < indices32UC1.rows; ++y )
        for ( int x = 0; x < indices32UC1.cols; ++x )
        {
            std::cout << "("
                      << y << ","
                      << x << ","
                      << indices32UC1.at<unsigned>(y,x) << ","
                      <<   indices[1].at<unsigned>(y,x) << ","
                      << "  " << indices32UC1.at<unsigned>(y,x)/3
                      << ");";
        }
    std::cout << std::endl;

    cv::Mat indices2;
    indices32UC1.convertTo( indices2, CV_32FC1, 600000.f );
    cv::imshow( "indicesFC1", indices2 );
    {
        double minVal, maxVal;
        cv::minMaxIdx( indices2, &minVal, &maxVal );
        std::cout << "minVal(indices2): " << minVal << ", "
                  << "maxVal(indices2): " << maxVal << std::endl;
    }


    cv::Mat out;
    depthFC1.convertTo( out, CV_16UC1 );
    cv::imwrite( "distances.png", out );

    {
        double minVal, maxVal;
        cv::minMaxIdx( depthFC1, &minVal, &maxVal );
        std::cout << "minVal(depth): " << minVal << ", "
                  << "maxVal(depth): " << maxVal << std::endl;
    }

    cv::Mat indicesFC1( indices32UC1.rows, indices32UC1.cols, CV_32FC1 );
    for ( int y = 0; y < indicesFC1.rows; ++y )
    {
        for ( int x = 0; x < indicesFC1.cols; ++x )
        {
            //indicesFC1.at<float>( y, x ) = ((float)*((unsigned*)indices32UC1.ptr<uchar>(y, x * 4)));
            //indicesFC1.at<float>( y, x ) = ((float)*((unsigned*)indices32UC1.ptr<uchar>(y, x * 4)));
            indicesFC1.at<float>( y, x ) = indices32UC1.at<float>( y, x );
        }
    }

    {
        double minVal, maxVal;
        cv::minMaxIdx( indicesFC1, &minVal, &maxVal );
        std::cout << "minVal(indicesFC1): " << minVal << ", "
                  << "maxVal(indicesFC1): " << maxVal << std::endl;
    }

    cv::imshow( "indices", indicesFC1 / 530679.f );

    cv::Mat indices16UC1;
    indicesFC1.convertTo( indices16UC1, CV_16UC1 );
    cv::imwrite( "incidex.png", indices16UC1 );

    cv::waitKey();

    return(0);
}

