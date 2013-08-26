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
#include <pcl/io/vtk_lib_io.h>
#include <eigen3/Eigen/Dense>

struct GLRenderer
{
        GLRenderer()
            : width(1280), height(960),
              pixels( new GLubyte[width*height*sizeof(float   ) * 4] ),
              ids   ( new GLuint [width*height*sizeof(unsigned) * 3] ),
              depthrenderbuffer(0), FramebufferName(0), renderedTexture({-1,-1})
        {}

        ~GLRenderer()
        {
            if ( pixels ) { delete [] pixels; pixels = NULL; }
            if ( ids    ) { delete [] ids   ; ids    = NULL; }

            glDeleteTextures( 2, renderedTexture );
            glDeleteFramebuffers(1, &FramebufferName);
            glDeleteRenderbuffers(1, &depthrenderbuffer);
        }

        int W() { return width; }
        int H() { return height; }
        GLubyte *& Pixels() { return pixels; }
        GLuint  *& Ids   () { return ids;    }
        int SzPixels() { return width * height * sizeof(float   ) * 4; }
        int SzIds()    { return width * height * sizeof(unsigned) * 3; }

        GLuint depthrenderbuffer;
        GLuint FramebufferName;
        GLuint renderedTexture[2];

        Mesh meshes;

    protected:
        int width;
        int height;
        GLubyte *pixels;
        GLuint  *ids;
        int szPixels;

} gInst;


// Shader Names
const char *vertexFileName   = "../src/shaders/triangles.vert";
const char *fragmentFileName = "../src/shaders/triangles.frag";

// Program and Shader Identifiers
GLuint p,v,f;

// Vertex Attribute Locations
GLuint vertexLoc, texLoc, normalLoc;

// Uniform variable Locations
GLuint projMatrixLoc, viewMatrixLoc, modelMatrixLoc, eyeLoc;

// Vertex Array Objects Identifiers
GLuint vao[3];


// storage for Matrices
float projMatrix[16];
float viewMatrix[16];
float modelMatrix[16];
float eyePosition[3];

// ----------------------------------------------------
// VECTOR STUFF
//

// res = a cross b;
void crossProduct( float *a, float *b, float *res) {

    res[0] = a[1] * b[2]  -  b[1] * a[2];
    res[1] = a[2] * b[0]  -  b[2] * a[0];
    res[2] = a[0] * b[1]  -  b[0] * a[1];
}

// Normalize a vec3
void normalize(float *a) {

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
void setIdentityMatrix( float *mat, int size) {

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
void multMatrix(float *a, float *b) {

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
void setTranslationMatrix(float *mat, float x, float y, float z) {

    setIdentityMatrix(mat,4);
    mat[12] = x;
    mat[13] = y;
    mat[14] = z;
}

// ----------------------------------------------------
// Projection Matrix
//

void buildProjectionMatrix(float fov, float ratio, float nearP, float farP)
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

void setCamera( float posX, float posY, float posZ,
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

    viewMatrix[0]  = right[0];
    viewMatrix[4]  = right[1];
    viewMatrix[8]  = right[2];
    viewMatrix[12] = 0.0f;

    viewMatrix[1]  = -up[0];
    viewMatrix[5]  = -up[1];
    viewMatrix[9]  = -up[2];
    viewMatrix[13] = 0.0f;

    viewMatrix[2]  = -dir[0];
    viewMatrix[6]  = -dir[1];
    viewMatrix[10] = -dir[2];
    viewMatrix[14] =  0.0f;

    viewMatrix[3]  = 0.0f;
    viewMatrix[7]  = 0.0f;
    viewMatrix[11] = 0.0f;
    viewMatrix[15] = 1.0f;

    setTranslationMatrix(aux, -posX, -posY, -posZ);

    multMatrix(viewMatrix, aux);

    eyePosition[0] = posX;
    eyePosition[1] = posY;
    eyePosition[2] = posZ;

    setIdentityMatrix( modelMatrix, 4 );
}

// ----------------------------------------------------

void changeSize(int w, int h) {

    float ratio;
    // Prevent a divide by zero, when window is too short
    // (you cant make a window of zero width).
    if(h == 0)
        h = 1;

    // Set the viewport to be the entire window
    glViewport(0, 0, w, h);

    //ratio = (1.0f * w) / h;
    //buildProjectionMatrix(53.13f, ratio, 1.0f, 30.0f);
}

void setupBuffers( int width, int height )
{
    //glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32UI, WindowWidth, WindowHeight, 0, GL_RGB_INTEGER, GL_UNSIGNED_INT, NULL);
    //glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, m_pickingTexture, 0);

    glGenFramebuffers( 1, &gInst.FramebufferName );
    glBindFramebuffer(GL_FRAMEBUFFER, gInst.FramebufferName );

    {
        glGenTextures( 2, gInst.renderedTexture );
        // "Bind" the newly created texture : all future texture functions will modify this texture
        glBindTexture( GL_TEXTURE_2D, gInst.renderedTexture[0] );
        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, gInst.renderedTexture[0], 0 );
    }

    // Create the texture object for the primitive information buffer
    {
        glBindTexture(GL_TEXTURE_2D, gInst.renderedTexture[1] );
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB32UI, width, height, 0, GL_RGB_INTEGER, GL_UNSIGNED_INT, NULL );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
        glFramebufferTexture2D (GL_DRAW_FRAMEBUFFER, GL_COLOR_ATTACHMENT1, GL_TEXTURE_2D, gInst.renderedTexture[1], 0 );
        //glReadPixels(x, y, 1, 1, GL_RGB_INTEGER, GL_UNSIGNED_INT, &Pixel);
    }

    // The depth buffer
    {
        glGenRenderbuffers(1, &gInst.depthrenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, gInst.depthrenderbuffer);
        glRenderbufferStorage( GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height );
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, gInst.depthrenderbuffer);

        // Create the texture object for the depth buffer
        //glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, WindowWidth, WindowHeight, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
        //glFramebufferTexture2D(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, m_depthTexture, 0);
    }

    // Set the list of draw buffers.
    GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0, GL_COLOR_ATTACHMENT1 };
    glDrawBuffers( 2, DrawBuffers ); // "1" is the size of DrawBuffers

    // Always check that our framebuffer is ok
    if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
        std::cerr << "asdf" << std::endl;
}

void setUniforms() {

    // must be called after glUseProgram
    glUniformMatrix4fv(projMatrixLoc,  1, false,  projMatrix );
    glUniformMatrix4fv(viewMatrixLoc,  1, false,  viewMatrix );
    glUniformMatrix4fv(modelMatrixLoc, 1, false, modelMatrix );
    glUniform3f( eyeLoc, eyePosition[0], eyePosition[1], eyePosition[2] );
}

void renderScene( void )
{
    // Render to our framebuffer
    //setCamera( 10,2,8,0,2,-5);

//    Eigen::Affine3f pose; pose.linear() << 1.f, 0.f, 0.f,
//                                  0.f, 1.f, 0.f,
//                                  0.f, 0.f, 1.f;
//    pose.translation() << 0.f, 0.f, -0.3f;
    Eigen::Affine3f pose; pose.linear() << 1.f, 0.f, 0.f,
                                            0.f, 1.f, 0.f,
                                            0.f, 0.f, 1.f;
    pose.translation() << 1.5f, 1.5f, -0.3f;


    Eigen::Vector3f pos_vector     = pose * Eigen::Vector3f (0, 0, 0);
    Eigen::Vector3f look_at_vector = pose.rotation () * Eigen::Vector3f (0, 0, 1) + pos_vector;
    Eigen::Vector3f up_vector      = pose.rotation () * Eigen::Vector3f (0, -1, 0);

    std::cout << "pos_vector: " << pos_vector << std::endl;

    setCamera(  pos_vector[0], pos_vector[1], pos_vector[2],
                look_at_vector[0], look_at_vector[1], look_at_vector[2],
                up_vector[0], up_vector[1], up_vector[2] );
    //setCamera( 10,2,8,0,2,-5,0,1,0);


    glUseProgram(p);
    setUniforms();

    std::cout << "current projMat: " << std::endl;
    for ( int y = 0; y < 4; ++y )
    {
        for (int x = 0; x < 4; ++x )
        {
            std::cout << projMatrix[y*4+x] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    std::cout << "current viewMat: " << std::endl;
    for ( int y = 0; y < 4; ++y )
    {
        for (int x = 0; x < 4; ++x )
        {
            std::cout << viewMatrix[y*4+x] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    glBindFramebuffer( GL_FRAMEBUFFER, gInst.FramebufferName );
    glDrawBuffers( 2, gInst.renderedTexture );
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glViewport( 0, 0, gInst.W(), gInst.H() ); // Render on the whole framebuffer, complete from the lower left corner to the upper right

    gInst.meshes.Render();

    //glFinish();

    //glBindFramebuffer( GL_FRAMEBUFFER, FramebufferName );
    //glReadBuffer( GL_COLOR_ATTACHMENT0 );
    //glReadPixels(0, 0, glRendererInstance.W(), glRendererInstance.H(), GL_RGBA, GL_FLOAT, glRendererInstance.Pixels() );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    //glutSwapBuffers();

    // disable
    GLenum tmpBuff[] = { GL_COLOR_ATTACHMENT0 };
    glDrawBuffers( 1, tmpBuff );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glClear( GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT );

    gInst.meshes.Render();

    //glFinish();
    glutSwapBuffers();
}

void processNormalKeys( unsigned char key, int x, int y )
{
    if ( key == 27 )
    {
        glDeleteVertexArrays(3,vao);
        glDeleteProgram(p);
        glDeleteShader(v);
        glDeleteShader(f);
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

GLuint setupShaders() {

    char *vs = NULL,*fs = NULL,*fs2 = NULL;

    GLuint p,v,f;

    v = glCreateShader( GL_VERTEX_SHADER   );
    f = glCreateShader( GL_FRAGMENT_SHADER );

    vs = textFileRead( vertexFileName   );
    fs = textFileRead( fragmentFileName );

    const char * vv = vs;
    const char * ff = fs;

    glShaderSource( v, 1, &vv, NULL );
    glShaderSource( f, 1, &ff, NULL );

    free(vs);free(fs);

    glCompileShader(v);
    glCompileShader(f);

    printShaderInfoLog(v);
    printShaderInfoLog(f);

    p = glCreateProgram();
    glAttachShader(p,v);
    glAttachShader(p,f);

    glBindFragDataLocation( p, 0, "outputF" );
    glBindFragDataLocation( p, 1, "ids"     );
    glLinkProgram(p);
    printProgramInfoLog(p);

    vertexLoc = glGetAttribLocation( p, "position" );
    std::cout << "vertexLoc: " << vertexLoc << std::endl;

    texLoc    = glGetAttribLocation( p, "texCoord" );
    std::cout << "texLoc: " << texLoc << std::endl;

    normalLoc  = glGetAttribLocation( p, "normal"   );
    std::cout << "colorLoc: " << normalLoc << std::endl;

    projMatrixLoc = glGetUniformLocation(p, "projMatrix");
    viewMatrixLoc = glGetUniformLocation(p, "viewMatrix");
    modelMatrixLoc = glGetUniformLocation(p, "modelMatrix");
    eyeLoc = glGetUniformLocation(p, "eyePos");

    return(p);
}

int main( int argc, char **argv )
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize( gInst.W(), gInst.H() );
    glutCreateWindow("Lighthouse 3D");

    glutDisplayFunc(renderScene);
    glutIdleFunc(renderScene);
    glutReshapeFunc(changeSize);
    glutKeyboardFunc(processNormalKeys);

    glewInit();
    if (glewIsSupported("GL_VERSION_3_3"))
        printf("Ready for OpenGL 3.3\n");
    else {
        printf("OpenGL 3.3 not supported\n");
        exit(1);
    }

    glEnable(GL_DEPTH_TEST);
    glClearColor(0.0,0.0,0.0,1.0);

    p = setupShaders();
    setupBuffers( gInst.W(), gInst.H() );

    gInst.meshes.LoadMesh( "/home/bontius/workspace/rec/testing/cloud_mesh.ply" );
    gInst.meshes.vertexShaderLoc    = vertexLoc;
    gInst.meshes.texShaderLoc       = texLoc;
    gInst.meshes.normalShaderLoc    = normalLoc;

    Eigen::Matrix3f intrinsics;
    intrinsics << 521.7401 * 2.f, 0             , 323.4402 * 2.f,
                  0             , 522.1379 * 2.f, 258.1387 * 2.f,
                  0             , 0             , 1             ;
    double width  = 1280.;
    double height = 960.;
    double znear  = 0.001;
    double zfar   = 10.01;
    int x0 = 0;
    int y0 = 0;
    Eigen::Matrix3f K(intrinsics);

    float ratio = (1.0f * width) / height;
    buildProjectionMatrix(53.13f, ratio, 0.001f, 10.01f);
    for ( int y = 0; y < 4; ++y )
    {
        for ( int x = 0; x < 4; ++x )
        {
            std::cout << projMatrix[y*4+x] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    float proj[16] = {  2.*K(0,0)/width,                      -2. * K(0,1) /  width,                                  0.,  0.,
                        0              ,                       2. * K(1,1) / height,                                  0.,  0.,
                        0              , (  width - 2. * K(0,2) + 2. * x0) /  width,    (-zfar - znear) / (zfar - znear), -1.,
                        0              , (-height + 2. * K(1,2) + 2. * y0) / height, -2. * zfar * znear / (zfar - znear),  0. };
    /*float proj[16] = { 2.*K(0,0)/width, -2. * K(0,1) / width , (width  - 2. * K(0,2) + 2. * x0)/width ,                            0,
                        0              , -2. * K(1,1) / height, (height - 2. * K(1,2) + 2. * y0)/height,                            0,
                        0              ,                     0, (-zfar - znear)/(zfar - znear)         , -2*zfar*znear/(zfar - znear),
                        0              ,                     0,                                      -1,                            0 };*/
    memcpy( projMatrix, proj, 16*sizeof(float) );

    for ( int y = 0; y < 4; ++y )
    {
        for ( int x = 0; x < 4; ++x )
        {
            std::cout << projMatrix[y*4+x] << ",";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    glutMainLoopEvent();
    //glutMainLoop();

    glBindFramebuffer( GL_FRAMEBUFFER, gInst.FramebufferName );
    glReadBuffer( GL_COLOR_ATTACHMENT0 );
    glReadPixels(0, 0, gInst.W(), gInst.H(), GL_RGBA, GL_FLOAT, gInst.Pixels() );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    cv::Mat ca0( gInst.H(), gInst.W(), CV_32FC4, gInst.Pixels() );
    std::vector<cv::Mat> chans;
    cv::split( ca0, chans );
    cv::imshow( "chan3", chans[3] );
    cv::imshow( "chan2", chans[2] );
    cv::imshow( "chan1", chans[1] );
    cv::imshow( "chan0", chans[0] );

    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[3], &minVal, &maxVal );
        std::cout << "minVal(chans[3]): " << minVal << ", "
                  << "maxVal(chans[3]): " << maxVal << std::endl;
    }
    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[2], &minVal, &maxVal );
        std::cout << "minVal(chans[2]): " << minVal << ", "
                  << "maxVal(chans[2]): " << maxVal << std::endl;
    }
    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[1], &minVal, &maxVal );
        std::cout << "minVal(chans[1]): " << minVal << ", "
                  << "maxVal(chans[1]): " << maxVal << std::endl;
    }
    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[0], &minVal, &maxVal );
        std::cout << "minVal(chans[0]): " << minVal << ", "
                  << "maxVal(chans[0]): " << maxVal << std::endl;
    }


    bool printed = false;
    for ( int i = 0; i < gInst.SzPixels() / 4; i+= 16 )
    {
        int val = static_cast<int>(
                      round(
                          reinterpret_cast<float*>(gInst.Pixels())[i] * 10.f
                          )
                      );
        if ( val > 0 )
        {
            printed = true;

            int val2 = static_cast<int>(
                                  round(
                                      reinterpret_cast<float*>(gInst.Pixels())[i+2]
                                      )
                                  );
            std::cout << "(" << reinterpret_cast<float*>(gInst.Pixels())[i] * 10.f << ","
                      << val << "," << val2 << ")";
        }
    }
    if ( printed ) std::cout << std::endl;

    glBindFramebuffer( GL_FRAMEBUFFER, gInst.FramebufferName );
    glReadBuffer( GL_COLOR_ATTACHMENT1 );
    glReadPixels(0, 0, gInst.W(), gInst.H(), GL_RGB_INTEGER, GL_UNSIGNED_INT, gInst.Ids() );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    cv::Mat ca1( gInst.H(), gInst.W(), CV_32SC3, gInst.Ids() );
    cv::split( ca1, chans );
    cv::imshow( "ca1C2", chans[2] );
    cv::imshow( "ca1C1", chans[1] );
    cv::imshow( "ca1C0", chans[0] );

    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[2], &minVal, &maxVal );
        std::cout << "minVal(chans[2]): " << minVal << ", "
                  << "maxVal(chans[2]): " << maxVal << std::endl;
    }
    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[1], &minVal, &maxVal );
        std::cout << "minVal(chans[1]): " << minVal << ", "
                  << "maxVal(chans[1]): " << maxVal << std::endl;
    }
    {
        double minVal, maxVal;
        cv::minMaxIdx( chans[0], &minVal, &maxVal );
        std::cout << "minVal(chans[0]): " << minVal << ", "
                  << "maxVal(chans[0]): " << maxVal << std::endl;
    }
    cv::waitKey();

    return(0);
}
