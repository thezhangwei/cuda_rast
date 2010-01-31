/*
 * Copyright 1993-2009 NVIDIA Corporation.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property and 
 * proprietary rights in and to this software and related documentation and 
 * any modifications thereto.  Any use, reproduction, disclosure, or distribution 
 * of this software and related documentation without an express license 
 * agreement from NVIDIA Corporation is strictly prohibited.
 * 
 */

/* Template project which demonstrates the basics on how to setup a project 
 * example application.
 * Device code.
 */

#ifndef _TEMPLATE_KERNEL_H_
#define _TEMPLATE_KERNEL_H_

#include <stdio.h>

#define SDATA( index)      cutilBankChecker(sdata, index)

__global__ void naivRaster_kernel(bool* buff, float* e1, float* e2, float* e3, float* p1, float* p2, float* p3){
	const unsigned int xcoord = blockDim.x*blockIdx.x+threadIdx.x;
	const unsigned int ycoord = blockDim.y*blockIdx.y+threadIdx.y;
	const float pixelx = (float)xcoord+0.5f;
	const float pixely = (float)ycoord+0.5f;
	const unsigned int idx = ycoord*blockDim.x*gridDim.x + xcoord;
	

	float tmp1,tmp2,tmp3;
	
	tmp1 = e1[0]*(pixelx-p1[0])+e1[1]*(pixely-p1[1]);
	tmp2 = e2[0]*(pixelx-p2[0])+e2[1]*(pixely-p2[1]);
	tmp3 = e3[0]*(pixelx-p3[0])+e3[1]*(pixely-p3[1]);
	
	
	if(tmp1 < 0.0f && tmp2 < 0.0f && tmp3 < 0.0f){
		buff[idx] = true;
	}else{
		buff[idx] = false;
	}
	
	//buff[idx] = (tmp1 < 0.0f && tmp2 < 0.0f && tmp3 < 0.0f);

}

__global__ void naivRaster2_kernel(bool* buff, float* data){
	const unsigned int xcoord = blockDim.x*blockIdx.x+threadIdx.x;
	const unsigned int ycoord = blockDim.y*blockIdx.y+threadIdx.y;
	const float pixelx = (float)xcoord+0.5f;
	const float pixely = (float)ycoord+0.5f;
	const unsigned int idx = ycoord*blockDim.x*gridDim.x + xcoord;
	const unsigned int bid = blockDim.x*threadIdx.y+threadIdx.x;
	
	__shared__ float sdata[12];
	float tmp1,tmp2,tmp3;
	
	
	if(bid < 12){
		sdata[bid] = data[bid];
	}
	__syncthreads();
	
	tmp1 = sdata[0]*(pixelx-sdata[6])+sdata[1]*(pixely-sdata[7]);
	tmp2 = sdata[2]*(pixelx-sdata[8])+sdata[3]*(pixely-sdata[9]);
	tmp3 = sdata[4]*(pixelx-sdata[10])+sdata[5]*(pixely-sdata[11]);
	
	if(tmp1 < 0.0f && tmp2 < 0.0f && tmp3 < 0.0f){
		buff[idx] = true;
	}else{
		buff[idx] = false;
	}

	//buff[idx] = (tmp1 < 0.0f && tmp2 < 0.0f && tmp3 < 0.0f);
}

__global__ void recursRaster_kernel(bool* buff, float* data){
	const unsigned int xcoord = (blockDim.x*blockIdx.x + threadIdx.x)*4;
	const unsigned int ycoord = (blockDim.y*blockIdx.y + threadIdx.y)*4;

	const float tlxcoord = (float)xcoord;
	const float tlycoord = (float)ycoord;
	const float trxcoord = (float)xcoord + 4.0f;
	const float trycoord = (float)ycoord;
	const float blxcoord = (float)xcoord;
	const float blycoord = (float)ycoord + 4.0f;
	const float brxcoord = (float)xcoord + 4.0f;
	const float brycoord = (float)ycoord + 4.0f;

	const unsigned int xpixelcoord = blockDim.x*blockIdx.x*4 + (threadIdx.z % 4)*4 + threadIdx.x;
	const unsigned int ypixelcoord = blockDim.y*blockIdx.y*4 + (threadIdx.z / 4)*4 + threadIdx.y;
	const float xpixel = (float)xpixelcoord + 0.5f;
	const float ypixel = (float)ypixelcoord + 0.5f;
	
	const unsigned int tmpid = threadIdx.y*4+threadIdx.x;

	__shared__ bool tmpdata[16];
	__shared__ float sdata[12];
	
	
	float tmp1,tmp2,tmp3;
	
	
	if(threadIdx.z == 0 && tmpid < 12){
		sdata[tmpid] = data[tmpid];	
	}
	
	__syncthreads();

	if(threadIdx.z == 0){

		
		tmp1 = sdata[0]*(brxcoord-sdata[6])+sdata[1]*(brycoord-sdata[7]);
		tmp2 = sdata[2]*(blxcoord-sdata[8])+sdata[3]*(blycoord-sdata[9]);
		tmp3 = sdata[4]*(tlxcoord-sdata[10])+sdata[5]*(tlycoord-sdata[11]);
		
		tmpdata[threadIdx.y*4+threadIdx.x] = (tmp1 < 0.0f && tmp2 < 0.0f && tmp3 < 0.0f);
		


	}
	
	__syncthreads();
	
	if(tmpdata[threadIdx.z]){
		
		
		tmp1 = sdata[0]*(xpixel-sdata[6])+sdata[1]*(ypixel-sdata[7]);
		tmp2 = sdata[2]*(xpixel-sdata[8])+sdata[3]*(ypixel-sdata[9]);
		tmp3 = sdata[4]*(xpixel-sdata[10])+sdata[5]*(ypixel-sdata[11]);
		
		if(tmp1 < 0.0f && tmp2 < 0.0f && tmp3 < 0.0f){
			buff[ypixelcoord*64+xpixelcoord] = true;
		}

	}



}
////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
__global__ void
testKernel( float* g_idata, float* g_odata) 
{
  // shared memory
  // the size is determined by the host application
  extern  __shared__  float sdata[];

  // access thread id
  const unsigned int tid = threadIdx.x;
  // access number of threads in this block
  const unsigned int num_threads = blockDim.x;

  // read in input data from global memory
  // use the bank checker macro to check for bank conflicts during host
  // emulation
  SDATA(tid) = g_idata[tid];
  __syncthreads();

  // perform some computations
  SDATA(tid) = (float) num_threads * SDATA( tid);
  __syncthreads();

  // write data to global memory
  g_odata[tid] = SDATA(tid);
}

#endif // #ifndef _TEMPLATE_KERNEL_H_
