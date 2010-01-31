
__global__ void transform_kernel( float4* outpos, float4* inpos, unsigned int width, unsigned int height, float* mvp_matrix, float* vp_matrix)
{
    // Indices into the VBO data. Roughly like texture coordinates from GLSL.
    unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int bx = threadIdx.x;
	unsigned int by = threadIdx.y;

    // TODO :: Write some value to pos[y*width + x], and make a particle system!
	/*outpos[ty*width + tx].x = inpos[ty*width + tx].x ;
	outpos[ty*width + tx].y = inpos[ty*width + tx].y ;
	outpos[ty*width + tx].z = inpos[ty*width + tx].z ;
	outpos[ty*width + tx].w = inpos[ty*width + tx].w ;*/

	////////////////////////////////////////////////////////////////////////////
	// shared memory
	////////////////////////////////////////////////////////////////////////////
	__shared__ float4 coords[16][16];
	__shared__ float mvps[16];
	__shared__ float vps[16];

	mvps[bx] = mvp_matrix[bx];
	vps[bx] = vp_matrix[bx];
	////////////////////////////////////////////////////////////////////////////
	// matrix transformation: modelview -> projection
	////////////////////////////////////////////////////////////////////////////
	coords[bx][by].x =	  mvps[0] * inpos[ty*width + tx].x
						+ mvps[1] * inpos[ty*width + tx].y
						+ mvps[2] * inpos[ty*width + tx].z
						+ mvps[3] * inpos[ty*width + tx].w;

	coords[bx][by].y =	  mvps[4] * inpos[ty*width + tx].x
						+ mvps[5] * inpos[ty*width + tx].y
						+ mvps[6] * inpos[ty*width + tx].z
						+ mvps[7] * inpos[ty*width + tx].w;

	coords[bx][by].z =	  mvps[8] * inpos[ty*width + tx].x
						+ mvps[9] * inpos[ty*width + tx].y
						+ mvps[10] * inpos[ty*width + tx].z
						+ mvps[11] * inpos[ty*width + tx].w;

	coords[bx][by].w =	  mvps[12] * inpos[ty*width + tx].x
						+ mvps[13] * inpos[ty*width + tx].y
						+ mvps[14] * inpos[ty*width + tx].z
						+ mvps[15] * inpos[ty*width + tx].w;
	
	__syncthreads();

	////////////////////////////////////////////////////////////////////////////
	// normalization
	////////////////////////////////////////////////////////////////////////////
	coords[bx][by].x = coords[bx][by].x / coords[bx][by].w;
	coords[bx][by].y = coords[bx][by].y / coords[bx][by].w;
	coords[bx][by].z = coords[bx][by].z / coords[bx][by].w;
	coords[bx][by].w = coords[bx][by].w / coords[bx][by].w;

	__syncthreads();

	////////////////////////////////////////////////////////////////////////////
	// matrix transformation: viewport
	////////////////////////////////////////////////////////////////////////////
	outpos[ty*width + tx].x =	  vps[0] * coords[bx][by].x
								+ vps[1] * coords[bx][by].y
								+ vps[2] * coords[bx][by].z
								+ vps[3] * coords[bx][by].w;

	outpos[ty*width + tx].y =	  vps[4] * coords[bx][by].x
								+ vps[5] * coords[bx][by].y
								+ vps[6] * coords[bx][by].z
								+ vps[7] * coords[bx][by].w;

	outpos[ty*width + tx].z =	  vps[8] * coords[bx][by].x
								+ vps[9] * coords[bx][by].y
								+ vps[10] * coords[bx][by].z
								+ vps[11] * coords[bx][by].w;

	outpos[ty*width + tx].w =	  vps[12] * coords[bx][by].x
								+ vps[13] * coords[bx][by].y
								+ vps[14] * coords[bx][by].z
								+ vps[15] * coords[bx][by].w;
	
	//__syncthreads();

}


/*

__global__ void tiangle_setup_kernel(float4* outpos, float4* verts, unsigned int width, unsigned int height, float* mvp_matrix, float* vp_matrix)
{
	// Indices into the VBO data. Roughly like texture coordinates from GLSL.
    unsigned int tx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int ty = blockIdx.y*blockDim.y + threadIdx.y;
	unsigned int bx = threadIdx.x;
	unsigned int by = threadIdx.y;

	// Shared memory
	__shared__ float4 coord1[8][8];
	__shared__ float4 coord2[8][8];
	__shared__ float4 coord3[8][8];
	//__shared__ float signed_area;
	__shared__ float3 coefs[8][8];

	// copy vertices to shared memory
	coord1[bx][by] = verts[3*ty*width + 3*tx];
	coord2[bx][by] = verts[3*ty*width + 3*tx + 1];
	coord3[bx][by] = verts[3*ty*width + 3*tx + 2];

	__syncthreads();
	
	// area
	float signed_area =  (coord2[bx][by].x - coord1[bx][by].x)*(coord3[bx][by].y - coord1[bx][by].y)
						-(coord3[bx][by].x - coord1[bx][by].x)*(coord2[bx][by].y - coord1[bx][by].y);

	__syncthreads();

	// coefs
	coefs[bx][by].x = coord1[bx][by].y - coord2[bx][by].y;
	coefs[bx][by].y = coord2[bx][by].x - coord1[bx][by].x;
	if(signed_area < 0)
	{	
		coefs[bx][by].x = -coefs[bx][by].x;
		coefs[bx][by].y = -coefs[bx][by].y;
	}

	__syncthreads();
	coefs[bx][by].z = -coefs[bx][by].x*coord1[bx][by].x - coefs[bx][by].y*coord1[bx][by].y;

}

*/