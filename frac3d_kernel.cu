/*
 * Lab 6 - Volume Rendering Fractals
 */

#ifndef _TEXTURE3D_KERNEL_H_
#define _TEXTURE3D_KERNEL_H_

#include "cutil_math.h"

/* Volume texture declaration */
texture<float, 3, cudaReadModeElementType> tex;

typedef struct {
    float4 m[3];
} float3x4;

__constant__ float3x4 c_invViewMatrix;  // inverse view matrix

/* Need to write host code to set these */
__constant__ float4 c_juliaC; // julia set constant
__constant__ float4 c_juliaPlane; // plane eqn of 3D slice


struct Ray {
	float3 o;	// origin
	float3 d;	// direction
};


// multiply two quaternions
__device__ float4
mul_quat(float4 p, float4 q)
{
    return make_float4(p.x*q.x-p.y*q.y-p.z*q.z-p.w*q.w,
		       p.x*q.y+p.y*q.x+p.z*q.w-p.w*q.z,
		       p.x*q.z-p.y*q.w+p.z*q.x+p.w*q.y,
		       p.x*q.w+p.y*q.z-p.z*q.y+p.w*q.x);
}

// square a quaternion (could be optimized)
__device__ float4
sqr_quat(float4 p)
{
    // this could/should be optimized
    return mul_quat(p,p);
}

// convert a 3d position to a 4d quaternion using plane-slice
__device__ float4
pos_to_quat(float3 pos, float4 plane)
{
    return make_float4(pos.x, pos.y, pos.z,
		       plane.x*pos.x+plane.y*pos.y+plane.z*pos.z+plane.w);
}

// intersect ray with a box
// http://www.siggraph.org/education/materials/HyperGraph/raytrace/rtinter3.htm

__device__
int intersectBox(Ray r, float3 boxmin, float3 boxmax, float *tnear, float *tfar)
{
    // compute intersection of ray with all six bbox planes
    float3 invR = make_float3(1.0f) / r.d;
    float3 tbot = invR * (boxmin - r.o);
    float3 ttop = invR * (boxmax - r.o);

    // re-order intersections to find smallest and largest on each axis
    float3 tmin = fminf(ttop, tbot);
    float3 tmax = fmaxf(ttop, tbot);

    // find the largest tmin and the smallest tmax
    float largest_tmin = fmaxf(fmaxf(tmin.x, tmin.y), fmaxf(tmin.x, tmin.z));
    float smallest_tmax = fminf(fminf(tmax.x, tmax.y), fminf(tmax.x, tmax.z));

    *tnear = largest_tmin;
    *tfar = smallest_tmax;

    return smallest_tmax > largest_tmin;
}

// transform vector by matrix (no translation)
__device__
float3 mul(const float3x4 &M, const float3 &v)
{
    float3 r;
    r.x = dot(v, make_float3(M.m[0]));
    r.y = dot(v, make_float3(M.m[1]));
    r.z = dot(v, make_float3(M.m[2]));
    return r;
}

// transform vector by matrix with translation
__device__
float4 mul(const float3x4 &M, const float4 &v)
{
    float4 r;
    r.x = dot(v, M.m[0]);
    r.y = dot(v, M.m[1]);
    r.z = dot(v, M.m[2]);
    r.w = 1.0f;
    return r;
}

// color conversion functions
__device__ uint rgbaFloatToInt(float4 rgba)
{
    rgba.x = __saturatef(rgba.x);   // clamp to [0.0, 1.0]
    rgba.y = __saturatef(rgba.y);
    rgba.z = __saturatef(rgba.z);
    rgba.w = __saturatef(rgba.w);
    return (uint(rgba.w*255)<<24) | (uint(rgba.z*255)<<16) | (uint(rgba.y*255)<<8) | uint(rgba.x*255);
}

__device__ uint rFloatToInt(float r)
{
    r = __saturatef(r);   // clamp to [0.0, 1.0]
    return (uint(r*255)<<24) | (uint(r*255)<<16) | (uint(r*255)<<8) | uint(r*255);
}

// get a normal from volume texture
/* feel free to use this, but you should also compute the normal
   using JuliaDist */
__device__ float3 d_TexNormal(float3 pos)
{
    float3 normal = make_float3(0);
    float d = 0.04f;

    normal.x = (tex3D(tex, pos.x+d, pos.y, pos.z)-tex3D(tex, pos.x-d, pos.y, pos.z));
    normal.y = (tex3D(tex, pos.x, pos.y+d, pos.z)-tex3D(tex, pos.x, pos.y-d, pos.z));
    normal.z = (tex3D(tex, pos.x, pos.y, pos.z+d)-tex3D(tex, pos.x, pos.y, pos.z-d));

    return normalize(normal);
}


// computes julia distance function
__device__ float
d_JuliaDist(float3 pos, int niter)
{
    int i;

//    float4 z0 = pos_to_quat(pos, c_juliaPlane);
//    float4 zp = make_float4(1,0,0,0);

    /* TODO: fill in JuliaDist function */

    return 0.0f;
}

// perform volume rendering
__global__ void
d_render(uint *d_output, uint imageW, uint imageH, float epsilon)
{
    // amount to step by
    float tstep = 0.0015f;
    int maxSteps = 2000;
    
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    float u = (x / (float) imageW)*2.0f-1.0f;
    float v = (y / (float) imageH)*2.0f-1.0f;

    // calculate eye ray in world space
    Ray eyeRay;
    eyeRay.o = make_float3(mul(c_invViewMatrix, make_float4(0.0f, 0.0f, 0.0f, 1.0f)));
    eyeRay.d = normalize(make_float3(u, v, -2.0f));
    eyeRay.d = mul(c_invViewMatrix, eyeRay.d);

    // find intersection with box
    float tnear, tfar;

    // return if not intersecting
    if (!intersectBox(eyeRay, 
		      make_float3(-2.0f,-2.0f,-2.0f), 
		      make_float3(2.0f,2.0f,2.0f),
		      &tnear, &tfar))
	return;

    // clamp to near plane
    if (tnear < 0.0f) tnear = 0.0f;

    float t = tnear;

    // accumulate values
    float accum = 0.0f;
    
    // start stepping through space
    for(int i=0; i<maxSteps; i++) {		
        pos = eyeRay.o + eyeRay.d*t;
        
        // map position to [0, 1] coordinates
        pos = pos*0.25f+0.5f;    

        // read from 3D texture
        float sample = tex3D(tex, pos.x, pos.y, pos.z);

	accum += sample;

        t += tstep;

        if (t > tfar) break;
    }

    /* TODO: calculate normal vector */

    if ((x < imageW) && (y < imageH)) {
        // write output color
        uint i = __umul24(y, imageW) + x;

	float4 col4 = make_float4(accum*0.01f);

	/* TODO: calculate output color based on lighting and position */

        d_output[i] = rgbaFloatToInt(col4);
    }
}

// recompute julia set at a single volume point
__global__ void
d_setfractal(float *d_output)
{
    // get x,y,z indices from kernel
//    uint x = threadIdx.x;
    ulong i = 0;

    /* TODO: get y, z coordinates from blockIdx,
       compute juliadist at position */

    // set output value
    d_output[i] = 0.0f;
}

#endif // #ifndef _TEXTURE3D_KERNEL_H_
