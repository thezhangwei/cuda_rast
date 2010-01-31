/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>
#include "cutil_math.h"

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



__global__ void triangle_setup_kernel( uint* o_buff, float4* i_verts, unsigned int triNum, unsigned int pntNum)
{
	//////////////////////////
	// Indices 
	//////////////////////////
    unsigned int tx = blockIdx.x*blockDim.x; //tile index.x
    unsigned int ty = blockIdx.y*blockDim.y; //tile index.y
	unsigned int px = threadIdx.x; // pixel index.x
	unsigned int py = threadIdx.y; // pixel index.y
	unsigned int idx = threadIdx.x*blockDim.x + threadIdx.y; // thread index

	//////////////////////////
	// Shared memory
	//////////////////////////
	__shared__ float4 p1[16];
	__shared__ float4 p2[16];
	__shared__ float4 p3[16];
	__shared__ float3 c1[16];
	__shared__ float3 c2[16];
	__shared__ float3 c3[16];
	__shared__ float signed_area[16];
	__shared__ bool rej[16];
	__shared__ bool acp[16];
	__shared__ uint s_buff[16][16];

	//////////////////////////
	// initialize s_buff
	//////////////////////////
	s_buff[px][py] = 0;
	
	
	//////////////////////////
	// load vertices
	//////////////////////////
	//float4* i_vert2 = (float4*)((char*)i_verts + pitch);
	//float4* i_vert3 = (float4*)((char*)i_verts + 2*pitch);

	//unsigned int i = 0;
	//for(unsigned int i = 0; i < triNum; i++)
	//{
		// fetch interleaving array 16 tris at a time
		//p1[px] = i_verts[px + i*blockDim.x];
		//p2[px] = i_verts[px + i*blockDim.x + triNum];
		//p3[px] = i_verts[px + i*blockDim.x + 2*triNum];
		
		//if(px == 0 && py == 0)
		//{
			// fetch continuous array 1 tri at a time 
			/*p1[px] = i_verts[i*3];
			p2[px] = i_verts[i*3 + 1];
			p3[px] = i_verts[i*3 + 2];

			__syncthreads();
	
			//////////////////////////
			// area of the triangle CCW positive !!
			//////////////////////////
			signed_area[px] =	(p2[px].x - p1[px].x)*(p3[px].y - p1[px].y) - 
								(p3[px].x - p1[px].x)*(p2[px].y - p1[px].y);

			//////////////////////////
			// setup edge func coefs !!! + inside !!!
			//////////////////////////
			c1[px].x = p1[px].y - p2[px].y;
			c1[px].y = p2[px].x - p1[px].x;
			c2[px].x = p2[px].y - p3[px].y;
			c2[px].y = p3[px].x - p2[px].x;
			c3[px].x = p3[px].y - p1[px].y;
			c3[px].y = p1[px].x - p3[px].x;

			if(signed_area < 0)
			{
				c1[px].x = -c1[px].x;
				c1[px].y = -c1[px].y;
				c2[px].x = -c2[px].x;
				c2[px].y = -c2[px].y;
				c3[px].x = -c3[px].x;
				c3[px].y = -c3[px].y;
			}

			//c1[px].z = - c1[px].x*p1[px].x - c1[px].y*p1[px].y;
			//c2[px].z = - c2[px].x*p2[px].x - c2[px].y*p2[px].y;
			//c3[px].z = - c3[px].x*p3[px].x - c3[px].y*p3[px].y;
		}

		__syncthreads();
		//////////////////////////
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
		}

		
		// whole tile inside
		if(py < 4 && px < 4)
		{
			acp[py*4 + px] = (	accept(tx + 4*py, ty + 4*px, blockDim.x/4, blockDim.y/4, p1[0], c1[0]) &&
								accept(tx + 4*py, ty + 4*px, blockDim.x/4, blockDim.y/4, p2[0], c2[0]) &&
								accept(tx + 4*py, ty + 4*px, blockDim.x/4, blockDim.y/4, p3[0], c3[0]));
			if(px == 0 && py == 0) acp[py*4 + px] = true;
			else acp[py*4 + px] = false;
		}
		__syncthreads();*/
		if(true)
		{
			// update the buffer to be 255s
			float4 rgba = {0.0f,0.0f,1.0f,0.0f};
			s_buff[py%4 *4 + px%4][(py/4) *4 + (int)(px/4)] = rgbaFloatToInt(rgba);
			//break;
		}
		__syncthreads();


		
	//}
	
	//////////////////////////
	// copy shared buffer back to global memory
	//////////////////////////
	o_buff[(py)*16 + 0 + px] = s_buff[px][py];
	
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
