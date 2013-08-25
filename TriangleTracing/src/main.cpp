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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct GLRenderer
{
        GLRenderer()
            : width(320), height(320),
              pixels( new GLubyte[width*height*sizeof(float) * 4] )
        {}

        ~GLRenderer()
        {
            if ( pixels ) { delete [] pixels; pixels = NULL; }
        }

        int W() { return width; }
        int H() { return height; }
        GLubyte *& Pixels() { return pixels; }
        int SzPixels() { return width * height * sizeof(float) * 4; }

    protected:
        int width;
        int height;
        GLubyte *pixels;
        int szPixels;

} glRendererInstance;

// Data for drawing Axis
float verticesAxis[] = {-20.0f, 0.0f, 0.0f, 1.0f,
            20.0f, 0.0f, 0.0f, 1.0f,

            0.0f, -20.0f, 0.0f, 1.0f,
            0.0f,  20.0f, 0.0f, 1.0f,

            0.0f, 0.0f, -20.0f, 1.0f,
            0.0f, 0.0f,  20.0f, 1.0f};

float colorAxis[] = {   0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f,
            0.0f, 0.0f, 0.0f, 0.0f};

// Data for triangle 1
float vertices1[] = { -3.0f, 0.0f, -5.0f, 1.0f,
                      -1.0f, 0.0f, -5.0f, 1.0f,
                      -2.0f, 2.0f, -5.0f, 1.0f,
                       1.f, 1.f, 1.f, 1.f,
                       2.f, 2.f, 2.f, 1.f,
                       1.f, 2.f, 2.f, 1.f,
                      1.0f, 0.0f, -5.0f, 1.0f,
                      3.0f, 0.0f, -5.0f, 1.0f,
                      2.0f, 2.0f, -5.0f, 1.0f };

float colors1[] = { 1.0f, 0.0f, 1.0f, 1.0f,
                    2.0f, 0.0f, 1.0f, 1.0f,
                    3.0f, 0.0f, 1.0f, 1.0f,
                    4.0f, 0.0f, 1.0f, 1.0f,
                    5.0f, 0.0f, 1.0f, 1.0f,
                    6.0f, 0.0f, 1.0f, 1.0f,
                    7.0f, 0.0f, 1.0f, 1.0f,
                    8.0f, 0.0f, 1.0f, 1.0f,
                    9.0f, 0.0f, 1.0f, 1.0f };
unsigned int faceArray [] = { 0, 1, 2,
                              1, 2, 3,
                              3, 4, 5,
                              6, 7, 8 };

// Data for triangle 2
float vertices2[] = {   1.0f, 0.0f, -5.0f, 1.0f,
            3.0f, 0.0f, -5.0f, 1.0f,
            2.0f, 2.0f, -5.0f, 1.0f};

float colors2[] = { 1.0f, 0.0f, 0.0f, 1.0f,
            1.0f, 0.0f, 0.0f, 1.0f,
            1.0f,0.0f, 0.0f, 1.0f};

// Shader Names
char *vertexFileName = "../src/shaders/triangles.vert";
char *fragmentFileName = "../src/shaders/triangles.frag";

// Program and Shader Identifiers
GLuint p,v,f;

// Vertex Attribute Locations
GLuint vertexLoc, colorLoc;

// Uniform variable Locations
GLuint projMatrixLoc, viewMatrixLoc, modelMatrixLoc;

// Vertex Array Objects Identifiers
GLuint vao[3];
GLuint FramebufferName = 0;
GLuint renderedTexture = 0;

// storage for Matrices
float projMatrix[16];
float viewMatrix[16];
float modelMatrix[16];

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

void buildProjectionMatrix(float fov, float ratio, float nearP, float farP) {

    float f = 1.0f / tan (fov * (M_PI / 360.0));

    setIdentityMatrix(projMatrix,4);

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

void setCamera(float posX, float posY, float posZ,
               float lookAtX, float lookAtY, float lookAtZ) {

    float dir[3], right[3], up[3];

    up[0] = 0.0f;   up[1] = 1.0f;   up[2] = 0.0f;

    dir[0] =  (lookAtX - posX);
    dir[1] =  (lookAtY - posY);
    dir[2] =  (lookAtZ - posZ);
    normalize(dir);

    crossProduct(dir,up,right);
    normalize(right);

    crossProduct(right,dir,up);
    normalize(up);

    float aux[16];

    viewMatrix[0]  = right[0];
    viewMatrix[4]  = right[1];
    viewMatrix[8]  = right[2];
    viewMatrix[12] = 0.0f;

    viewMatrix[1]  = up[0];
    viewMatrix[5]  = up[1];
    viewMatrix[9]  = up[2];
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

    ratio = (1.0f * w) / h;
    buildProjectionMatrix(53.13f, ratio, 1.0f, 30.0f);
}

void setupBuffers() {

    GLuint buffers[3];

    glGenVertexArrays(3, vao);
    //
    // VAO for first triangle
    //
    glBindVertexArray(vao[0]);
    // Generate two slots for the vertex and color buffers
    glGenBuffers(3, buffers);
    // bind buffer for vertices and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices1), vertices1, GL_STATIC_DRAW);
    glEnableVertexAttribArray(vertexLoc);
    glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, 0, 16, 0);

    // bind buffer for colors and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors1), colors1, GL_STATIC_DRAW);
    glEnableVertexAttribArray(colorLoc);
    glVertexAttribPointer(colorLoc, 4, GL_FLOAT, 0, 16, 0);

    // buffer for faces

    //glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffers[2]);
    //glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(faceArray), faceArray, GL_STATIC_DRAW );

    //
    // VAO for second triangle
    //
    glBindVertexArray(vao[1]);
    // Generate two slots for the vertex and color buffers
    glGenBuffers(2, buffers);

    // bind buffer for vertices and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices2), vertices2, GL_STATIC_DRAW);
    glEnableVertexAttribArray(vertexLoc);
    glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, 0, 0, 0);

    // bind buffer for colors and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colors2), colors2, GL_STATIC_DRAW);
    glEnableVertexAttribArray(colorLoc);
    glVertexAttribPointer(colorLoc, 4, GL_FLOAT, 0, 0, 0);

    //
    // This VAO is for the Axis
    //
    glBindVertexArray(vao[2]);
    // Generate two slots for the vertex and color buffers
    glGenBuffers(2, buffers);
    // bind buffer for vertices and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[0]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(verticesAxis), verticesAxis, GL_STATIC_DRAW);
    glEnableVertexAttribArray(vertexLoc);
    glVertexAttribPointer(vertexLoc, 4, GL_FLOAT, 0, 0, 0);

    // bind buffer for colors and copy data into buffer
    glBindBuffer(GL_ARRAY_BUFFER, buffers[1]);
    glBufferData(GL_ARRAY_BUFFER, sizeof(colorAxis), colorAxis, GL_STATIC_DRAW);
    glEnableVertexAttribArray(colorLoc);
    glVertexAttribPointer(colorLoc, 4, GL_FLOAT, 0, 0, 0);

    // The framebuffer, which regroups 0, 1, or more textures, and 0 or 1 depth buffer.

    {
        glGenFramebuffers( 1, &FramebufferName );
        glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
        // The texture we're going to render to

        glGenTextures(1, &renderedTexture);

        // "Bind" the newly created texture : all future texture functions will modify this texture
        glBindTexture(GL_TEXTURE_2D, renderedTexture);

        // Give an empty image to OpenGL ( the last "0" )
        glTexImage2D( GL_TEXTURE_2D, 0, GL_RGB, 320, 320, 0, GL_RGBA, GL_UNSIGNED_BYTE, 0);

        // Poor filtering. Needed !
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        // The depth buffer
        GLuint depthrenderbuffer;
        glGenRenderbuffers(1, &depthrenderbuffer);
        glBindRenderbuffer(GL_RENDERBUFFER, depthrenderbuffer);
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, 320, 320);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthrenderbuffer);

        // Set "renderedTexture" as our colour attachement #0
        glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, renderedTexture, 0);

        // Set the list of draw buffers.
        GLenum DrawBuffers[2] = {GL_COLOR_ATTACHMENT0};
        glDrawBuffers(1, DrawBuffers); // "1" is the size of DrawBuffers

        // Always check that our framebuffer is ok
        if ( glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cerr << "asdf" << std::endl;

    }

}

void setUniforms() {

    // must be called after glUseProgram
    glUniformMatrix4fv(projMatrixLoc,  1, false, projMatrix );
    glUniformMatrix4fv(viewMatrixLoc,  1, false, viewMatrix );
    glUniformMatrix4fv(modelMatrixLoc,  1, false, modelMatrix );
}

void renderScene(void) {

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    setCamera( 10,2,8,0,2,-5);

    glUseProgram(p);
    setUniforms();

    glBindVertexArray( vao[0] );
    glDrawArrays( GL_TRIANGLES, 0, sizeof(vertices1) / 16 );
    //glDrawElements( GL_TRIANGLES, sizeof(faceArray), GL_UNSIGNED_INT, 0 );

    // Render to our framebuffer
    glBindFramebuffer(GL_FRAMEBUFFER, FramebufferName);
    glViewport(0,0,glRendererInstance.W(),glRendererInstance.H()); // Render on the whole framebuffer, complete from the lower left corner to the upper right

    glFinish();

    //glBindFramebuffer( GL_FRAMEBUFFER, FramebufferName );
    //glReadBuffer( GL_COLOR_ATTACHMENT0 );
    //glReadPixels(0, 0, glRendererInstance.W(), glRendererInstance.H(), GL_RGBA, GL_FLOAT, glRendererInstance.Pixels() );
    //glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    glBindVertexArray(vao[2]);
    glDrawArrays(GL_LINES, 0, 6);

    glutSwapBuffers();
}

void processNormalKeys(unsigned char key, int x, int y) {

    if (key == 27) {
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

    v = glCreateShader(GL_VERTEX_SHADER);
    f = glCreateShader(GL_FRAGMENT_SHADER);

    vs = textFileRead(vertexFileName);
    fs = textFileRead(fragmentFileName);

    const char * vv = vs;
    const char * ff = fs;

    glShaderSource(v, 1, &vv,NULL);
    glShaderSource(f, 1, &ff,NULL);

    free(vs);free(fs);

    glCompileShader(v);
    glCompileShader(f);

    printShaderInfoLog(v);
    printShaderInfoLog(f);

    p = glCreateProgram();
    glAttachShader(p,v);
    glAttachShader(p,f);

    glBindFragDataLocation(p, 0, "outputF");
    glLinkProgram(p);
    printProgramInfoLog(p);

    vertexLoc = glGetAttribLocation(p,"position");
    colorLoc = glGetAttribLocation(p, "color");

    projMatrixLoc = glGetUniformLocation(p, "projMatrix");
    viewMatrixLoc = glGetUniformLocation(p, "viewMatrix");
    modelMatrixLoc = glGetUniformLocation(p, "modelMatrix");

    return(p);
}

int main(int argc, char **argv) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DEPTH | GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowPosition(100,100);
    glutInitWindowSize( glRendererInstance.W(), glRendererInstance.H() );
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
    setupBuffers();

    glutMainLoopEvent();
    //glutMainLoop();

    glBindFramebuffer( GL_FRAMEBUFFER, FramebufferName );
    glReadBuffer( GL_COLOR_ATTACHMENT0 );
    glReadPixels(0, 0, glRendererInstance.W(), glRendererInstance.H(), GL_RGBA, GL_FLOAT, glRendererInstance.Pixels() );
    glBindFramebuffer( GL_FRAMEBUFFER, 0 );

    bool printed = false;
    for ( int i = 0; i < glRendererInstance.SzPixels() / 4; i+= 16 )
    {
        int val = static_cast<int>(
                      round(
                          reinterpret_cast<float*>(glRendererInstance.Pixels())[i] * 10.f
                          )
                      );
        if ( val > 0 )
        {
            printed = true;

            int val2 = static_cast<int>(
                                  round(
                                      reinterpret_cast<float*>(glRendererInstance.Pixels())[i+2]
                                      )
                                  );
            std::cout << "(" << val << "," << val2 << ")";
        }
    }
    if ( printed ) std::cout << std::endl;
    cv::Mat m( glRendererInstance.H(), glRendererInstance.W(), CV_32FC4, glRendererInstance.Pixels() );
    std::vector<cv::Mat> chans;
    cv::split( m, chans );
    cv::imshow( "chan0", chans[0] / 10.f );
    cv::imshow( "chan3", chans[3] / 10.f );
    cv::waitKey();


    return(0);
}
