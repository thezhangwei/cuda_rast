/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "cutil_math.h"

// texture 
float4* d_triSrc;

texture<float4, 1, cudaReadModeElementType> tex_triSrc;

#define SDATA( index)      cutilBankChecker(sdata, index)

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
// rejection test return true if the tile is rejected !!
///////////////////////////////////////////////////////////////////////////////
__device__ bool reject(const unsigned int offx, const unsigned int offy, 
							const unsigned int tilew, const unsigned int tileh,
							float4 point, float3 coef)
{
	// right top corner
	if(coef.x >= 0 && coef.y >= 0)
	{
		float eval = coef.x*(offx + tilew - point.x) + coef.y*(offy + tileh - point.y);
		if(eval < 0)
			return true;
		else 
			return false;
	}
	// left top corner
	else if(coef.x < 0 && coef.y >= 0)
	{
		float eval = coef.x*(offx - point.x) + coef.y*(offy + tileh - point.y);
		if(eval < 0)
			return true;
		else 
			return false;
	}
	// left bottom corner
	else if(coef.x < 0 && coef.y < 0)
	{
		float eval = coef.x*(offx - point.x) + coef.y*(offy - point.y);
		if(eval < 0)
			return true;
		else 
			return false;
	}
	//right bottom corner
	else// if(coef.x >= 0 && coef.y < 0)
	{
		float eval = coef.x*(offx + tilew - point.x) + coef.y*(offy - point.y);
		if(eval < 0)
			return true;
		else 
			return false;
	}
}

///////////////////////////////////////////////////////////////////////////////
// acceptance test return true if the tile is accepted !!
///////////////////////////////////////////////////////////////////////////////
__device__ bool accept(const unsigned int offx, const unsigned int offy, 
							const unsigned int tilew, const unsigned int tileh,
							float4 point, float3 coef)
{
	// left bottom corner
	if(coef.x >= 0 && coef.y >= 0)
	{
		float eval = coef.x*(offx - point.x) + coef.y*(offy - point.y);
		if(eval >= 0)
			return true;
		else 
			return false;
	}
	// right bottom corner
	else if(coef.x < 0 && coef.y >= 0)
	{
		float eval = coef.x*(offx + tilew - point.x) + coef.y*(offy - point.y);
		if(eval >= 0)
			return true;
		else 
			return false;
	}
	// right top corner
	else if(coef.x < 0 && coef.y < 0)
	{
		float eval = coef.x*(offx + tilew - point.x) + coef.y*(offy + tileh - point.y);
		if(eval >= 0)
			return true;
		else 
			return false;
	}
	// left top corner
	else //if(coef.x >= 0 && coef.y < 0)
	{
		float eval = coef.x*(offx - point.x) + coef.y*(offy + tileh - point.y);
		if(eval >= 0)
			return true;
		else 
			return false;
	}
}


__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}



__global__ void triangle_setup_kernel( uint* o_buff, float4* i_verts, unsigned int triNum, unsigned int vpt)
{
	//////////////////////////
	// Indices 
	//////////////////////////
	unsigned int px = blockIdx.x*blockDim.x + threadIdx.x; // pixel index x
	unsigned int py = blockIdx.y*blockDim.y + threadIdx.y; // pixel index y
	unsigned int tx = threadIdx.x; // thread index x in block
	unsigned int ty = threadIdx.y; // thread index y in block
	unsigned int idx = threadIdx.x*blockDim.x + threadIdx.y; // thread index
	//unsigned int triIdx = py*blockDim.x*gridDim.x + by + idx; // global index of the triangle array

	const float xpix = (float) px + 0.5f;
	const float ypix = (float) py + 0.5f;
	
	//////////////////////////
	// Shared memory
	//////////////////////////
	__shared__ float4	p1[16], 
						p2[16], 
						p3[16]; //768
	__shared__ float2	c1[16], 
						c2[16], 
						c3[16]; //384
	//__shared__ float signed_area[16];
	//__shared__ bool rej[16];
	//__shared__ bool acp[16];
	__shared__ uint s_buff[16][16][4]; //4096

	//////////////////////////
	// initialize s_buff
	//////////////////////////
	s_buff[tx][ty][0] = 0;
	s_buff[tx][ty][1] = 0;
	s_buff[tx][ty][2] = 0;
	s_buff[tx][ty][3] = 0;
	//rej[idx] = false;
	//acp[idx] = true;
	//////////////////////////
	// load vertices
	//////////////////////////
	//float4* i_vert2 = (float4*)((char*)i_verts + pitch);
	//float4* i_vert3 = (float4*)((char*)i_verts + 2*pitch);

	//fetch 16 tris at one time !!
	unsigned int triLeft = triNum;
	for(unsigned int cnt = 0; cnt < (triNum-1)/blockDim.x + 1; cnt++)
	{
		// fetch interleaving array tris at a time
		unsigned int triFch = (triLeft >= blockDim.x)? (blockDim.x) : (triLeft);
		//(a % b != 0) ? (a / b + 1) : (a / b);
		if((idx) < triFch){
			
			unsigned int itmp = idx;

			p1[itmp] = tex1Dfetch(tex_triSrc, cnt*blockDim.x*vpt + 3*idx);
			p2[itmp] = tex1Dfetch(tex_triSrc, cnt*blockDim.x*vpt + 3*idx + 1);
			p3[itmp] = tex1Dfetch(tex_triSrc, cnt*blockDim.x*vpt + 3*idx + 2);
			//p2[itmp] = i_verts[cnt*blockDim.x*vpt + 3*idx + 1];
			//p3[itmp] = i_verts[cnt*blockDim.x*vpt + 3*idx + 2];
		
			//////////////////////////
			// setup edge func coefs !!! + inside !!!
			//////////////////////////
			c1[itmp].x = p1[itmp].y - p2[itmp].y;
			c1[itmp].y = p2[itmp].x - p1[itmp].x;
			c2[itmp].x = p2[itmp].y - p3[itmp].y;
			c2[itmp].y = p3[itmp].x - p2[itmp].x;
			c3[itmp].x = p3[itmp].y - p1[itmp].y;
			c3[itmp].y = p1[itmp].x - p3[itmp].x;

			//////////////////////////
			// area of the triangle CCW positive !!
			//////////////////////////
			/*signed_area[itmp] =	(p2[itmp].x - p1[itmp].x)*(p3[itmp].y - p1[itmp].y) - 
								(p3[itmp].x - p1[itmp].x)*(p2[itmp].y - p1[itmp].y);

			if(signed_area[itmp] < 0)
			{
				c1[itmp].x = -c1[itmp].x;
				c1[itmp].y = -c1[itmp].y;
				c2[itmp].x = -c2[itmp].x;
				c2[itmp].y = -c2[itmp].y;
				c3[itmp].x = -c3[itmp].x;
				c3[itmp].y = -c3[itmp].y;
			}

			c1[itmp].z = - c1[itmp].x*p1[itmp].x - c1[itmp].y*p1[itmp].y;
			c2[itmp].z = - c2[itmp].x*p2[itmp].x - c2[itmp].y*p2[itmp].y;
			c3[itmp].z = - c3[itmp].x*p3[itmp].x - c3[itmp].y*p3[itmp].y;*/
			
		}

		__syncthreads();


		/*//////////////////////////
		// tile test 16X16
		//////////////////////////
		// whole tile outside
		if(px == 0 && py == 0)
		{
			rej[0] = (	reject(tx, ty, blockDim.x, blockDim.y, p1[0], c1[0]) &&
						reject(tx, ty, blockDim.x, blockDim.y, p2[0], c2[0]) &&
						reject(tx, ty, blockDim.x, blockDim.y, p3[0], c3[0]));
		}
		__syncthreads();
		if(false) //rej[0]
		{
			// leave the buffer untouched
			//break;
		}
		__syncthreads();
		// whole tile inside
		if(px == 0 && py == 0)
		{
			acp[0] = (	accept(tx, ty, blockDim.x, blockDim.y, p1[0], c1[0]) &&
						accept(tx, ty, blockDim.x, blockDim.y, p2[0], c2[0]) &&
						accept(tx, ty, blockDim.x, blockDim.y, p3[0], c3[0]));
		}
		__syncthreads();
		if(false) //acp[0]
		{
			// should update the whole buffer !!
			s_buff[px][py] = 1.0f;
			//break;
		}
		__syncthreads();

		//////////////////////////
		// tile test 4X4
		//////////////////////////
		// whole tile outside
		
		/*if(py < 4 && px < 4)
		{
			rej[py*4 + px] = (	reject(tx + 4*py, ty + 4*px, blockDim.x/4, blockDim.y/4, p1[0], c1[0]) &&
								reject(tx + 4*py, ty + 4*px, blockDim.x/4, blockDim.y/4, p2[0], c2[0]) &&
								reject(tx + 4*py, ty + 4*px, blockDim.x/4, blockDim.y/4, p3[0], c3[0]));
		}
		__syncthreads();
		if(false) //rej[py]
		{ 
			// leave the buffer untouched
			break;
		}*/

		
		// whole tile inside
		/*if(ty < 4 && tx < 4)
		{
			acp[ty*4 + tx] = (	accept(bx + 4*ty, by + 4*tx, blockDim.x/4, blockDim.y/4, p1[0], c1[0]) &&
								accept(bx + 4*ty, by + 4*tx, blockDim.x/4, blockDim.y/4, p2[0], c2[0]) &&
								accept(bx + 4*ty, by + 4*tx, blockDim.x/4, blockDim.y/4, p3[0], c3[0]));
			if(tx == 0 && ty == 0) acp[ty*4 + tx] = true;
			else acp[ty*4 + tx] = false;
		}
		__syncthreads();*/

		float r1, r2, r3;
		for(unsigned int i = 0; i < triFch; i++){
			
			r1 = c1[i].x * (xpix - p1[i].x) + c1[i].y * (ypix - p1[i].y);
			r2 = c2[i].x * (xpix - p2[i].x) + c2[i].y * (ypix - p2[i].y);
			r3 = c3[i].x * (xpix - p3[i].x) + c3[i].y * (ypix - p3[i].y);
		
			if(r1 > 0.0f && r2 > 0.0f && r3 > 0.0f)
			{
				// update the buffer to be 255s
				float4 rgba = {1.0f,0.0f,0.0f,0.0f};
				//s_buff[tx][ty][0] = rgbaFloatToInt(rgba);
			}

			r1 = c1[i].x * (xpix + 256 - p1[i].x) + c1[i].y * (ypix - p1[i].y);
			r2 = c2[i].x * (xpix + 256 - p2[i].x) + c2[i].y * (ypix - p2[i].y);
			r3 = c3[i].x * (xpix + 256 - p3[i].x) + c3[i].y * (ypix - p3[i].y);
		
			if(r1 > 0.0f && r2 > 0.0f && r3 > 0.0f)
			{
				// update the buffer to be 255s
				float4 rgba = {1.0f,0.0f,0.0f,0.0f};
				//s_buff[tx][ty][1] = rgbaFloatToInt(rgba);
			}

			r1 = c1[i].x * (xpix - p1[i].x) + c1[i].y * (ypix + 256 - p1[i].y);
			r2 = c2[i].x * (xpix - p2[i].x) + c2[i].y * (ypix + 256 - p2[i].y);
			r3 = c3[i].x * (xpix - p3[i].x) + c3[i].y * (ypix + 256 - p3[i].y);

			if(r1 > 0.0f && r2 > 0.0f && r3 > 0.0f)
			{
				// update the buffer to be 255s
				float4 rgba = {1.0f,0.0f,0.0f,0.0f};
				s_buff[tx][ty][2] = rgbaFloatToInt(rgba);
			}

			r1 = c1[i].x * (xpix + 256 - p1[i].x) + c1[i].y * (ypix + 256 - p1[i].y);
			r2 = c2[i].x * (xpix + 256 - p2[i].x) + c2[i].y * (ypix + 256 - p2[i].y);
			r3 = c3[i].x * (xpix + 256 - p3[i].x) + c3[i].y * (ypix + 256 - p3[i].y);
		
			if(r1 > 0.0f && r2 > 0.0f && r3 > 0.0f)
			{
				// update the buffer to be 255s
				float4 rgba = {1.0f,0.0f,0.0f,0.0f};
				s_buff[tx][ty][3] = rgbaFloatToInt(rgba);
			}
			__syncthreads();
		}

		triLeft = triLeft - blockDim.x;
	}

		__syncthreads();
	
	//////////////////////////
	// copy shared buffer back to global memory
	//////////////////////////
	//o_buff[px + py*gridDim.x*blockDim.x*2] = s_buff[tx][ty][0];
	//o_buff[gridDim.x*blockDim.x + px + py*gridDim.x*blockDim.x*2] = s_buff[tx][ty][1];
	o_buff[gridDim.x*blockDim.x*2*gridDim.x*blockDim.x + px + py*gridDim.x*blockDim.x*2] = s_buff[tx][ty][2];
	o_buff[gridDim.x*blockDim.x*2*gridDim.x*blockDim.x + gridDim.x*blockDim.x + px + py*gridDim.x*blockDim.x*2] = s_buff[tx][ty][3];
	//o_buff[511] = s_buff[tx][ty];
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
