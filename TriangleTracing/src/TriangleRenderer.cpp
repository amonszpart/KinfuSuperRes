#include "TriangleRenderer.h"

#include "textfile.h"

#define M_PI 3.14159265358979323846

#include <pcl/io/vtk_lib_io.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>
#include <iostream>


// --------------------------------------------------------------------------------------------------------------------
#define INVALID_OGL_VALUE (0xFFFFFFFF)

using namespace std;
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

// --------------------------------------------------------------------------------------------------------------------

namespace am
{
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
    void TriangleRenderer::renderDepthAndIndices( std::vector<cv::Mat> &depths, std::vector<cv::Mat> &indices,
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
        indices.resize(3);
        readDepthToFC1( depths[0], alpha, depths[1] );
        readIds( indices[0], indices[1], indices[2] );
        return;

        std::vector<unsigned> vxIds( indices[1].cols * indices[1].rows );
        for ( int y = 0; y < indices[1].rows; ++y )
        {
            for ( int x = 0; x < indices[1].cols; ++x )
            {
                vxIds[ y * indices[1].cols + x ] = indices[0].at<unsigned>(y,x);
            }
        }

        std::vector<unsigned> trIds    ( indices[1].cols * indices[1].rows );
        std::vector<unsigned> trIdsFlat( indices[2].cols * indices[2].rows );
        for ( int y = 0; y < indices[1].rows; ++y )
        {
            for ( int x = 0; x < indices[1].cols; ++x )
            {
                trIds[ y * indices[1].cols + x ] = indices[1].at<unsigned>(y,x);
            }
        }

        for ( int y = 0; y < indices[2].rows; ++y )
        {
            for ( int x = 0; x < indices[2].cols; ++x )
            {
                trIdsFlat[ y * indices[2].cols + x ] = indices[2].at<unsigned>(y,x);
            }
        }

        unsigned maxTriangleID = *std::max_element( trIds.begin(), trIds.end() ) + 1;
        std::cout << "maxTriangleID: " << *std::max_element( trIds.begin(), trIds.end() ) << std::endl;
        std::cout << "maxVertexID: " << *std::max_element( vxIds.begin(), vxIds.end() ) << std::endl;
        if ( maxTriangleID > 530700 )
        {
            std::cerr << "aborting..." << std::endl;
            return;
        }

        // count triangles
        unsigned *triangleCounts = new unsigned[ maxTriangleID ];
        std::fill( triangleCounts, triangleCounts + maxTriangleID, 0 );
        for ( auto trId : trIds )
        {
            ++triangleCounts[ trId ];
        }
        std::cout << "minTriangleCount: " << *std::min_element( triangleCounts+1, triangleCounts+maxTriangleID ) << std::endl;
        std::cout << "maxTriangleCount: " << *std::max_element( triangleCounts+1, triangleCounts+maxTriangleID ) << std::endl;
        unsigned maxTrCntID = std::distance( triangleCounts, std::max_element(triangleCounts+1, triangleCounts+maxTriangleID) );
        std::cout << "maxTrCntID: " << maxTrCntID << std::endl;
        std::cout << "triangleCounts[maxTrCntID]: " << triangleCounts[maxTrCntID] << std::endl;

        // get vertex id
        std::vector<unsigned> vertexList;
        std::vector<unsigned> faceList;
        std::vector<unsigned> flatFaceList;
        std::vector<cv::Vec2i> coordinates;
        for ( unsigned i = 0; i < trIds.size(); ++i )
        {
            if ( trIds[i] == maxTrCntID )
            {
                vertexList.push_back( vxIds[i] );
                faceList.push_back( trIds[i] );
                flatFaceList.push_back( trIdsFlat[i] );
                coordinates.push_back( cv::Vec2i(i/w, i%w) );
            }
        }

        for ( unsigned i = 0; i < vertexList.size(); ++i )
        {
            std::cout << "(" << vertexList[i] << "," << faceList[i] << "," << flatFaceList[i] << ","
                         << coordinates[i][0] << "," << coordinates[i][1]
                         << "," << depths[1].at<float>( coordinates[i][0], coordinates[i][1] ) * 1000000.f
                         << "); ";
        }
        std::cout << std::endl;

        int v0id = meshPtr->polygons[maxTrCntID].vertices[0];
        int v1id = meshPtr->polygons[maxTrCntID].vertices[1];
        int v2id = meshPtr->polygons[maxTrCntID].vertices[2];
        //float v0

        std::vector< std::vector<cv::Vec2i> > smalls(4);
        smalls[0].push_back( cv::Vec2i( 0, 0) );
        smalls[0].push_back( cv::Vec2i(-1, 0) );
        smalls[0].push_back( cv::Vec2i( 0,-1) );

        smalls[1].push_back( cv::Vec2i( 0, 0) );
        smalls[1].push_back( cv::Vec2i( 0,-1) );
        smalls[1].push_back( cv::Vec2i( 1, 0) );

        smalls[2].push_back( cv::Vec2i( 0, 0) );
        smalls[2].push_back( cv::Vec2i( 1, 0) );
        smalls[2].push_back( cv::Vec2i( 1, 1) );

        smalls[3].push_back( cv::Vec2i( 0, 0) );
        smalls[3].push_back( cv::Vec2i(-1, 1) );
        smalls[3].push_back( cv::Vec2i(-1, 0) );

       for ( int y = 294; y < 307; ++y )
        {
            for ( int x = 420; x < 433; ++x )
            {
                bool create = false;
                for ( int smid = 0; smid < smalls.size() && !create; ++smid )
                {
                    std::vector<cv::Vec2i> offs = smalls[smid];

                    if (    indices[1].at<unsigned>(y+offs[0][0],x+offs[0][1]) == indices[1].at<unsigned>(y+offs[1][0],x+offs[1][1])
                         && indices[1].at<unsigned>(y+offs[0][0],x+offs[0][1]) == indices[1].at<unsigned>(y+offs[2][0],x+offs[2][1]) )
                    {
                        create = true;
                    }
                }
                if ( create )
                    std::cout << indices[1].at<unsigned>(y,x) << " ";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        return;

        // get vertices of triangle counts > 3

        std::vector<std::vector<unsigned>> vertexlists( maxTriangleID );
        // for every triangle pixel
        for ( unsigned j = 0; j < trIds.size(); ++j )
        {
            unsigned trid = trIds[j];
            // if appears more then 3 times
            if ( triangleCounts[trid] > 3 )
            {
                //std::cout << "j: " << j << std::endl;
                //std::cout << "vxIds[j]: " << vxIds[j] << std::endl;
                fflush( stdout );
                // add vertex to this triangle list
                if ( (vertexlists[trid].size() == 0) ||
                     (std::find(vertexlists[trid].begin(), vertexlists[trid].end(), vxIds[j]) == vertexlists[trid].end()) )
                    vertexlists[trid].push_back( vxIds[j] );
            }
        }

        bool triples = false;
        for ( unsigned i = 0; i < vertexlists.size(); ++i )
        {
            if ( vertexlists[i].size() > 0 )
            {
                std::cout << "trid: " << i;
                for ( auto vxid : vertexlists[i] )
                {
                    std::cout << vxid << ",";
                }
                std::cout << std::endl;

                if ( vertexlists[i].size() > 3 )
                {
                    std::cerr << "asdfasdfasfasf" << std::endl;
                    triples = true;
                }
            }
        }
        if ( triples )
        {
            std::cerr << "wroooong" << std::endl;
        }
        else
            std::cerr << "OK" << std::endl;


        delete [] triangleCounts;
    }

    /*
     *\brief Loads mesh from path, and calls renderDepthAndIndices with the filled meshPtr
     */
    void TriangleRenderer::renderDepthAndIndices( std::vector<cv::Mat> &depths, std::vector<cv::Mat> &indices,
                                                  int w, int h, Eigen::Matrix3f const& intrinsics,
                                                  Eigen::Affine3f const& pose, std::string const& meshPath,
                                                  float alpha )
    {
        pcl::PolygonMesh::Ptr meshPtr( new pcl::PolygonMesh );
        pcl::io::loadPolygonFile( meshPath, *meshPtr );

        renderDepthAndIndices( depths, indices, w, h, intrinsics, pose, meshPtr, alpha );
    }

    /*
     *\brief    Read framebuffer object's distance data
     *\param[OUT] distances contains distances from camera in 0.f..alpha range
     *\param[IN]  alpha     scales distances
     *\param[OUT] indices   contains unusable (clamped) Ids // FIXME scale to 0..1.f by vertexcount in ".frag"
     */
    void TriangleRenderer::readDepthToFC1( cv::Mat &distances, float alpha, cv::Mat &indices )
    {
        const int pixels_point_step = sizeof(float) * 4;

        glBindFramebuffer( GL_FRAMEBUFFER, framebufferHandle_ );
        glReadBuffer( GL_COLOR_ATTACHMENT0 );
        glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE );
        glReadPixels( 0, 0, width_, height_, GL_RGBA, GL_FLOAT, pixels );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        distances.create( height_, width_, CV_32FC1 );
        indices.create( height_, width_, CV_32FC1 );
        for ( int y = 0; y < height_; ++y )
        {
            for ( int x = 0; x < width_; ++x )
            {
                // copy one channel only
                float val = *reinterpret_cast<float*>( &pixels[(y * width_ + x) * pixels_point_step ] ) * alpha;
                distances.at<float>( y, x ) = val;
                float ind = *reinterpret_cast<float*>( &pixels[(y * width_ + x) * pixels_point_step + sizeof(float)] );
                indices.at<float>( y, x) = ind;
            }
        }
    }

    void TriangleRenderer::setupBuffers( int width, int height )
    {
        if ( !inited_ ) { std::cerr << "GLRenderer::setupBuffers(): cannot run, buffers not inited!" << std::endl; return; }

        if (       textureHandles_[0] != INVALID_OGL_VALUE ) glDeleteTextures    ( 2,  textureHandles_          );
        if ( depthRenderBufferHandle_ != INVALID_OGL_VALUE ) glDeleteTextures    ( 1, &depthRenderBufferHandle_ );
        if (       framebufferHandle_ != INVALID_OGL_VALUE ) glDeleteFramebuffers( 1, &framebufferHandle_       );

        glGenFramebuffers( 1, &framebufferHandle_ );
        glBindFramebuffer(GL_FRAMEBUFFER, framebufferHandle_ );
        {
            glGenTextures( 3, textureHandles_ );
            glBindTexture( GL_TEXTURE_2D, textureHandles_[0] );

            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
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

        {
            glBindTexture( GL_TEXTURE_2D, textureHandles_[2] );

            glTexImage2D( GL_TEXTURE_2D, 0, GL_RGBA32F, width, height, 0, GL_RGBA, GL_FLOAT, 0);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
            glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT2, GL_TEXTURE_2D, textureHandles_[2], 0 );
        }

        // The depth buffer
        {
            glGenRenderbuffers(1, &depthRenderBufferHandle_);
            glBindRenderbuffer(GL_RENDERBUFFER, depthRenderBufferHandle_);
            glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );
            glFramebufferRenderbuffer( GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthRenderBufferHandle_ );
        }

        // Set the list of draw buffers.
        GLenum DrawBuffers[3] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1, GL_COLOR_ATTACHMENT2 };
        glDrawBuffers( 3, DrawBuffers ); // "1" is the size of DrawBuffers

        // Always check that our framebuffer is ok
        GLenum status = glCheckFramebufferStatus(GL_FRAMEBUFFER);
        if(status != GL_FRAMEBUFFER_COMPLETE)
        {
            std::cout << "Framebuffer: Incomplete framebuffer (";
            switch(status){
                case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
                    std::cout << "GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT";
                    break;
                case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
                    std::cout << "GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT";
                    break;
                case GL_FRAMEBUFFER_UNSUPPORTED:
                    std::cout << "GL_FRAMEBUFFER_UNSUPPORTED";
                    break;
            }
            std::cout << ")" << std::endl;
        }
    }


    /*
     *\brief    Read framebuffer object's vertex and triangle index data
     *\param[OUT] vertexIds    contains                vertex   indices as unsigned integers stored in a CV_8UC4 format, read with ".at<unsigned>(y,x)"
     *\param[OUT] triangleIds  contains interpolated   triangle indices as unsigned integers stored in a CV_8UC4 format, read with ".at<unsigned>(y,x)"
     *\param[OUT] triangleIds2 contains uninterpolated triangle indices as unsigned integers stored in a CV_8UC4 format, read with ".at<unsigned>(y,x)"
     */
    void TriangleRenderer::readIds( cv::Mat &vertexIds, cv::Mat &triangleIds, cv::Mat &triangleIds2 )
    {
        glBindFramebuffer( GL_FRAMEBUFFER, framebufferHandle_ );
        glReadBuffer( GL_COLOR_ATTACHMENT1 );
        glClampColor(GL_CLAMP_READ_COLOR, GL_FALSE );
        glReadPixels( 0, 0, width_, height_, GL_RGB_INTEGER, GL_UNSIGNED_INT, ids );
        glBindFramebuffer( GL_FRAMEBUFFER, 0 );

        vertexIds.create( height_, width_, CV_8UC4 );    // there's no CV_32UC1, so emulating it by CV_8UC4, minmaxloc won't be accurate
        triangleIds.create( height_, width_, CV_8UC4 );  // there's no CV_32UC1, so emulating it by CV_8UC4, minmaxloc won't be accurate
        triangleIds2.create( height_, width_, CV_8UC4 ); // there's no CV_32UC1, so emulating it by CV_8UC4, minmaxloc won't be accurate
        for ( int y = 0; y < height_; ++y )
        {
            for ( int x = 0; x < width_; ++x )
            {
                // copy one channel only

                vertexIds   .at<unsigned>(y,x) = ids[ (y * width_ + x) * 3     ]; // ids is GLuint pointer, so sizeof unsigned not needed
                triangleIds .at<unsigned>(y,x) = ids[ (y * width_ + x) * 3 + 1 ];
                triangleIds2.at<unsigned>(y,x) = ids[ (y * width_ + x) * 3 + 2 ];
            }
        }
    }


    void TriangleRenderer::setSize( int w, int h )
    {
        width_  = w;
        height_ = h;
        if ( pixels ) { delete [] pixels; pixels = NULL; }
        if ( ids    ) { delete [] ids   ; ids    = NULL; }

        pixels = new GLubyte[ width_*height_*sizeof(float   ) * 4 ];
        ids    = new GLuint [ width_*height_*sizeof(unsigned) * 3 ];

        this->setupBuffers( width_, height_ );
    }

    TriangleRenderer::TriangleRenderer()
        : width_(1280), height_(960),
          pixels( new GLubyte[width_*height_*sizeof(float   ) * 4] ),
          ids   ( new GLuint [width_*height_*sizeof(unsigned) * 3] ),
          depthRenderBufferHandle_( INVALID_OGL_VALUE ), framebufferHandle_( INVALID_OGL_VALUE ),
          inited_(false)
    {
        textureHandles_[0] = textureHandles_[1] = textureHandles_[2] = INVALID_OGL_VALUE;
        vertexFileName = "../../TriangleTracing/src/shaders/triangles.vert";
        fragmentFileName = "../../TriangleTracing/src/shaders/triangles.frag";
    }

    TriangleRenderer::~TriangleRenderer()
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

    void TriangleRenderer::init( int w, int h )
    {
        //Display *dpy;
        //dpy = XOpenDisplay(":0");

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

        glClampColor( GL_CLAMP_READ_COLOR, GL_FALSE );
        glClampColor( GL_CLAMP_VERTEX_COLOR, GL_FALSE );
        glClampColor( GL_CLAMP_FRAGMENT_COLOR, GL_FALSE );

        shader_program             = setupShaders();
        meshes_.vertexShaderLoc    = vertexLoc;
        meshes_.normalShaderLoc    = normalLoc;
        inited_ = true;

        this->setSize( w, h );
    }

    // "/home/bontius/workspace/rec/testing/cloud_mesh.ply"
    void TriangleRenderer::loadMesh( std::string const& mesh_path )
    {
        meshes_.loadMesh( mesh_path );
        meshes_.vertexShaderLoc    = vertexLoc;
        meshes_.normalShaderLoc    = normalLoc;
    }

    // View Matrix
    void TriangleRenderer::setCamera( float posX, float posY, float posZ,
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


    void TriangleRenderer::setUniforms()
    {
        // must be called after glUseProgram
        glUniformMatrix4fv(projMatrixLoc,  1, false,  projMatrix );
        glUniformMatrix4fv(viewMatrixLoc,  1, false,  viewMatrix_ );
        glUniformMatrix4fv(modelMatrixLoc, 1, false, modelMatrix_ );
        glUniform3f( eyeLoc, eyePosition_[0], eyePosition_[1], eyePosition_[2] );
    }

    void TriangleRenderer::setCamera( Eigen::Affine3f const& pose )
    {
        Eigen::Vector3f pos_vector     = pose * Eigen::Vector3f (0, 0, 0);
        Eigen::Vector3f look_at_vector = pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
        Eigen::Vector3f up_vector      = pose.rotation () * Eigen::Vector3f (0, -1, 0);

        setCamera( pos_vector[0], pos_vector[1], pos_vector[2],
                   look_at_vector[0], look_at_vector[1], look_at_vector[2],
                   up_vector[0], up_vector[1], up_vector[2] );
    }

    void TriangleRenderer::renderScene()
    {
        if ( !inited_                  ) { std::cerr << "GLRenderer::renderScene(): cannot run, buffers not inited!" << std::endl; return; }
        if ( !meshes_.NumberOfMeshes() ) { std::cerr << "GLRenderer::renderScene(): cannot run, no meshes!"          << std::endl; return; }

        glUseProgram( shader_program );
        setUniforms();

        // set render target
        glBindFramebuffer( GL_FRAMEBUFFER, framebufferHandle_ );
        glDrawBuffers( 3, textureHandles_ );
        glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );
        glClampColor( GL_CLAMP_READ_COLOR, GL_FALSE );
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

    GLuint TriangleRenderer::setupShaders()
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
        glBindFragDataLocation( shader_program, 2, "dists"     );
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

    void TriangleRenderer::setIntrinsics( Eigen::Matrix3f K, float zNear, float zFar )
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


} // end ns am
