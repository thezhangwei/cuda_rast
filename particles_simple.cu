#ifdef _WIN32

#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cutil.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>

#include "vecmath.h"

typedef unsigned int uint;
typedef unsigned char uchar;

////////////////////////////////////////////////////////////////////////////////
// constants
const unsigned int window_width = 512;
const unsigned int window_height = 512;

const unsigned int mesh_width = 48;
const unsigned int mesh_height = 48;

// Matrix
Mtx4f m;

// vbo variables
GLuint vbo;
GLuint vbo_out;
GLuint pbo;
GLuint mvpMatrix;
GLuint vpMatrix;

// Device buffer variables
float4* d_velocities;

float anim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_y = 0.0;
float translate_z = -3.0;

////////////////////////////////////////////////////////////////////////////////
// kernels
__global__ void transform_kernel( float4* outpos, float4* inpos, unsigned int width, unsigned int height, float* mvp_matrix, float* vp_matrix);
__global__ void triangle_setup_kernel( uint* o_buff, float4* i_verts, unsigned int triNum, unsigned int pntNum);

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runParticles( int argc, char** argv);

// GL functionality
CUTBoolean initGL();
void createVertexVBO( GLuint* vbo, GLuint* vbo_out);
void deleteVertexVBO( GLuint* vbo, GLuint* vbo_out);
void createMatrixVBO( GLuint* mvpMatrix, float* mvp_matrix, GLuint* vpMatrix, float* vp_matrix);
void deleteMatrixVBO( GLuint* mvpMatrix, GLuint* vpMatrix);
void initPixelBuffer( GLuint* pbo);

// rendering callbacks
void display();
void keyboard( unsigned char key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
void runCuda( GLuint vbo, GLuint vbo_out, GLuint pbo);
void checkResultCuda( int argc, char** argv, const GLuint& vbo);

void createDeviceData();
void deleteDeviceData();

// matrix helper function
void getViewportMatrix(Mtx4f &m);
void getModel2ScreenMatrix(Mtx4f &m);
void getScreen2ModelMatrix(Mtx4r &m);
void getScreen2ViewMatrix(Mtx4r &m);

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv)
{
    runParticles( argc, argv);

    CUT_EXIT(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple particle simulation for CUDA
////////////////////////////////////////////////////////////////////////////////
void runParticles( int argc, char** argv)
{
    CUT_DEVICE_INIT(argc, argv);

    // Create GL context
    glutInit( &argc, argv);
    glutInitDisplayMode( GLUT_RGBA | GLUT_DOUBLE);
    glutInitWindowSize( window_width, window_height);
    glutCreateWindow( "Simple Particles");

    // Init random number generator
    srand(time(0));

    // initialize GL
    if( CUTFalse == initGL()) {
        return;
    }

    // register callbacks
    glutDisplayFunc( display);
    glutKeyboardFunc( keyboard);
    glutMouseFunc( mouse);
    glutMotionFunc( motion);

    // create vertex VBO
    createVertexVBO( &vbo, &vbo_out);
	initPixelBuffer( &pbo);

    // create other device memory (velocities!)
    //createDeviceData();

    // start rendering mainloop
    glutMainLoop();
}

////////////////////////////////////////////////////////////////////////////////
//! Run the Cuda part of the computation
////////////////////////////////////////////////////////////////////////////////
void runCuda( GLuint vbo, GLuint vbo_out, GLuint pbo)
{
    // map OpenGL buffer object for writing from CUDA
    float4 *dptr;
	float4 *sptr;
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&sptr, vbo));
	CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&dptr, vbo_out));

	//get matrix: mvp, vp
	Mtx4f mv_matrix;
	Mtx4f pr_matrix;
	Mtx4f mvp_matrix;
	glGetFloatv(GL_MODELVIEW_MATRIX, mv_matrix.array);
	glGetFloatv(GL_PROJECTION_MATRIX, pr_matrix.array);
	mvp_matrix = pr_matrix * mv_matrix;

	Mtx4f vp_matrix;
	getViewportMatrix(vp_matrix);

	//create matrix vbo
	createMatrixVBO( &mvpMatrix, mvp_matrix.array, &vpMatrix, vp_matrix.array);
    
	float *mvpmptr;
	float *vpmptr;
	CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&mvpmptr, mvpMatrix));
    CUDA_SAFE_CALL(cudaGLMapBufferObject( (void**)&vpmptr, vpMatrix));

	//glGetFloatv(GL_MODELVIEW_MATRIX, m.array);
	//m.invert();

	//for(int i=0; i<16; i++)
	//	mvmatrix[i] = m.array[i];

	//initialize mvmatrix as a simple translation matrix

	// map PBO to get CUDA device pointer
    uint *d_output;
    CUDA_SAFE_CALL(cudaGLMapBufferObject((void**)&d_output, pbo));
    CUDA_SAFE_CALL(cudaMemset(d_output, 0, window_width*window_height*4));
	

    // execute the kernel
    dim3 block(16, 16, 1);
    dim3 grid(mesh_width / block.x, mesh_height / block.y, 1);
    transform_kernel<<< grid, block>>>( dptr, sptr, mesh_width, mesh_height, mvpmptr, vpmptr);

	dim3 block1(16, 16, 1);
    dim3 grid1(window_width / block.x, window_height / block.y, 1);
	triangle_setup_kernel<<<grid1, block1>>>( d_output, dptr, mesh_width*mesh_height, 3);

    // unmap buffer object
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject( mvpMatrix));
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vpMatrix));

	deleteMatrixVBO( &mvpMatrix, &vpMatrix);

	CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo));
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject( vbo_out));
	CUDA_SAFE_CALL(cudaGLUnmapBufferObject( pbo));
}

////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////
CUTBoolean initGL()
{
    // initialize necessary OpenGL extensions
    glewInit();
    if (! glewIsSupported( "GL_VERSION_2_0 " 
        "GL_ARB_pixel_buffer_object"
		)) {
        fprintf( stderr, "ERROR: Support for necessary OpenGL extensions missing.");
        fflush( stderr);
        return CUTFalse;
    }

    // default initialization
    glClearColor( 0.0, 0.0, 0.0, 1.0);
    glDisable( GL_DEPTH_TEST);

    // viewport
    glViewport( 0, 0, window_width, window_height);

    // projection
    glMatrixMode( GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(60.0, (GLfloat)window_width / (GLfloat) window_height, 0.1, 10.0);

    CUT_CHECK_ERROR_GL();

    return CUTTrue;
}

////////////////////////////////////////////////////////////////////////////////
//! Create Vertex VBO
////////////////////////////////////////////////////////////////////////////////
void createVertexVBO(GLuint* vbo, GLuint* vbo_out)
{
    // create buffer object
    glGenBuffers( 1, vbo);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo);

    // Initialize data.
    float4* temppos = (float4*)malloc(mesh_width*mesh_height*4*sizeof(float));
    for(int i = 0; i < mesh_width*mesh_height; ++i)
    {
        temppos[i].x = ((float)rand())/RAND_MAX;
        temppos[i].y = ((float)rand())/RAND_MAX;
        temppos[i].z = ((float)rand())/RAND_MAX;
        temppos[i].w = 1.;
    }

    // initialize buffer object
    unsigned int size = mesh_width * mesh_height * 4 * sizeof( float);
    glBufferData( GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);

    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo));

	//create vbo_out
	glGenBuffers( 1, vbo_out);
    glBindBuffer( GL_ARRAY_BUFFER, *vbo_out);
	for(int i = 0; i < mesh_width*mesh_height; ++i)
    {
        temppos[i].x = 0.0f;
        temppos[i].y = 0.0f;
        temppos[i].z = 0.0f;
        temppos[i].w = 0.0f;
    }
	glBufferData( GL_ARRAY_BUFFER, size, temppos, GL_DYNAMIC_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vbo_out));

    free(temppos);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete Vertex VBO
////////////////////////////////////////////////////////////////////////////////
void deleteVertexVBO( GLuint* vbo, GLuint* vbo_out)
{
    glBindBuffer( 1, *vbo);
    glDeleteBuffers( 1, vbo);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo));

    *vbo = 0;

	glBindBuffer( 1, *vbo_out);
    glDeleteBuffers( 1, vbo_out);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vbo_out));

    *vbo_out = 0;
}

////////////////////////////////////////////////////////////////////////////////
//! Create Matrix VBO
////////////////////////////////////////////////////////////////////////////////
void createMatrixVBO(GLuint* mvpMatrix, float* mvp_matrix, GLuint* vpMatrix, float* vp_matrix)
{
    // create buffer object
    glGenBuffers( 1, mvpMatrix);
    glBindBuffer( GL_ARRAY_BUFFER, *mvpMatrix);

    // Initialize data.
    //float* temp = (float*)malloc(16*sizeof(float));
	//temp = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};

    // initialize buffer object
    glBufferData( GL_ARRAY_BUFFER, 16*sizeof(float), mvp_matrix, GL_DYNAMIC_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0);

    // register buffer object with CUDA
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*mvpMatrix));

	//create vpMatrix
	glGenBuffers( 1, vpMatrix);
    glBindBuffer( GL_ARRAY_BUFFER, *vpMatrix);
	
	glBufferData( GL_ARRAY_BUFFER, 16*sizeof(float), vp_matrix, GL_DYNAMIC_DRAW);
    glBindBuffer( GL_ARRAY_BUFFER, 0);
	CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*vpMatrix));

    //free(temp);

    CUT_CHECK_ERROR_GL();
}

////////////////////////////////////////////////////////////////////////////////
//! Delete Matrix VBO
////////////////////////////////////////////////////////////////////////////////
void deleteMatrixVBO(GLuint* mvpMatrix, GLuint* vpMatrix)
{
    glBindBuffer( 1, *mvpMatrix);
    glDeleteBuffers( 1, mvpMatrix);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*mvpMatrix));

    *mvpMatrix = 0;

	glBindBuffer( 1, *vpMatrix);
    glDeleteBuffers( 1, vpMatrix);

    CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*vpMatrix));

    *vpMatrix = 0;
}
////////////////////////////////////////////////////////////////////////////////
//! Create PBO
////////////////////////////////////////////////////////////////////////////////
void initPixelBuffer(GLuint* pbo)
{
    /*if (&pbo) {
        // delete old buffer
        CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(*pbo));
        glDeleteBuffersARB(1, pbo);
    }*/

    // create pixel buffer object for display
    glGenBuffersARB(1, pbo);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, *pbo);
    glBufferDataARB(GL_PIXEL_UNPACK_BUFFER_ARB, window_width*window_height*sizeof(GLubyte)*4, 0, GL_STREAM_DRAW_ARB);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);
    
    CUDA_SAFE_CALL(cudaGLRegisterBufferObject(*pbo));

    // calculate new grid size
    //gridSize = dim3(iDivUp(window_width, blockSize.x), iDivUp(window_height, blockSize.y));
}

int iDivUp(int a, int b){
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

////////////////////////////////////////////////////////////////////////////////
//! Create device data
////////////////////////////////////////////////////////////////////////////////
void createDeviceData()
{
    // Create a velocity for every position.
    CUDA_SAFE_CALL( cudaMalloc( (void**)&d_velocities, mesh_width*mesh_height*
                                                    4 * sizeof(float) ) );

    // Initialize data.
    float4* tempvels = (float4*)malloc(mesh_width*mesh_height*4*sizeof(float));
    for(int i = 0; i < mesh_width*mesh_height; ++i)
    {
        tempvels[i].x = 2*((float)rand())/RAND_MAX-1.;
        tempvels[i].y = 2.*((float)rand())/RAND_MAX-1.;
        tempvels[i].z = 2.*((float)rand())/RAND_MAX-1.;
        tempvels[i].w = ((float)rand())/RAND_MAX;
    }

    // Copy to gpu
    CUDA_SAFE_CALL( cudaMemcpy( d_velocities, tempvels, mesh_width*mesh_height*4*sizeof(float), cudaMemcpyHostToDevice) );
    free(tempvels);
}

////////////////////////////////////////////////////////////////////////////////
//! Delete device data
////////////////////////////////////////////////////////////////////////////////
void deleteDeviceData()
{
    // Create a velocity for every position.
    CUDA_SAFE_CALL( cudaFree( d_velocities ) );
}

////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////
void display()
{
    // run CUDA kernel to generate vertex positions
    //runCuda(vbo, vbo_out);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // set view matrix
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(0.0, 0.0, translate_z);
    glRotatef(rotate_x, 1.0, 0.0, 0.0);
    glRotatef(rotate_y, 0.0, 1.0, 0.0);

	// run kernel after GL_MODELVIEW matrix has been set
	runCuda(vbo, vbo_out, pbo);

	//reset GL_MODELVIEW matrix to test transformation done in kernel
	//glMatrixMode(GL_MODELVIEW);
	//glPushMatrix();
    //glLoadIdentity();
    //glTranslatef(0.0, 0.0, -translate_z);
    //glRotatef(-rotate_x, 1.0, 0.0, 0.0);
    //glRotatef(-rotate_y, 0.0, 1.0, 0.0);

    // render from the vbo
    /*glBindBuffer(GL_ARRAY_BUFFER, vbo_out);
    glVertexPointer(4, GL_FLOAT, 0, 0);

    glEnableClientState(GL_VERTEX_ARRAY);
    glColor3f(1.0, 0.0, 0.0);
    glDrawArrays(GL_POINTS, 0, mesh_width * mesh_height);
    glDisableClientState(GL_VERTEX_ARRAY);

    glutSwapBuffers();
    glutPostRedisplay();*/

	// draw image from PBO
    glDisable(GL_DEPTH_TEST);
    glRasterPos2i(0, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, pbo);
    glDrawPixels(window_width, window_height, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glBindBufferARB(GL_PIXEL_UNPACK_BUFFER_ARB, 0);

    glutSwapBuffers();
    glutReportErrors();


    anim += 0.01;
}

////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard( unsigned char key, int /*x*/, int /*y*/)
{
    switch( key) {
    case( 27) :
        deleteVertexVBO( &vbo, &vbo_out);
		//deleteMatrixVBO( &mvpMatrix, &vpMatrix);
		CUDA_SAFE_CALL(cudaGLUnregisterBufferObject(pbo));    
		glDeleteBuffersARB(1, &pbo);
        //deleteDeviceData();
        exit( 0);
    }
}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////
void mouse(int button, int state, int x, int y)
{
    if (state == GLUT_DOWN) {
        mouse_buttons |= 1<<button;
    } else if (state == GLUT_UP) {
        mouse_buttons = 0;
    }

    mouse_old_x = x;
    mouse_old_y = y;
    glutPostRedisplay();
}

void motion(int x, int y)
{
    float dx, dy;
    dx = x - mouse_old_x;
    dy = y - mouse_old_y;

    if (mouse_buttons & 1) {
        rotate_x += dy * 0.2;
        rotate_y += dx * 0.2;
    } else if (mouse_buttons & 4) {
        translate_z += dy * 0.01;
    }

    mouse_old_x = x;
    mouse_old_y = y;
}


////////////////////////////////////////////////////////////////////////////////////////////
//Matrix helper functions
////////////////////////////////////////////////////////////////////////////////////////////
void getViewportMatrix(Mtx4f &m)
{
	int width = glutGet((GLenum)GLUT_WINDOW_WIDTH);
	int height = glutGet((GLenum)GLUT_WINDOW_WIDTH);
	float whalf = width*0.5f;
	float hhalf = height*0.5f;
	m.set(	whalf,	0,	0,	whalf,
			0,	hhalf,	0,	hhalf,
			0,	0,	0.5,	0.5,
			0,	0,	0,	1);
}

void getModel2ScreenMatrix(Mtx4f &m)
{
	Mtx4f mv, pr, vp; //modelView, projection, viewport
	glGetFloatv(GL_MODELVIEW_MATRIX, mv.array);
	glGetFloatv(GL_PROJECTION_MATRIX, pr.array);
	getViewportMatrix(vp);
	m = vp*(pr*mv);
}

void getScreen2ModelMatrix(Mtx4r &m)
{
	getModel2ScreenMatrix(m);
	bool result = m.invert();
	if(result == false)
	{
		printf("Err: reCamera -- could not invert Screen2Model matrix\n");
	}
}

void getScreen2ViewMatrix(Mtx4r &m)
{
	Mtx4f pr,vp; //projection, viewport
	glGetFloatv(GL_PROJECTION_MATRIX, pr.array);
	getViewportMatrix(vp);
	m = vp*(pr);

	bool result = m.invert();
	if(result == false)
	{
		printf("Err: reCamera -- could not invert Screen2View matrix\n");
	}
}
