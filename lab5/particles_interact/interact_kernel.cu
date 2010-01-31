

__device__ float3 particle_particle( float4 bi, float4 bj, float3 ai )
{
    // Hint, not strictly necessary (compute acceleration!)
}

__device__ float3 compute_accel(float4 pos, float4* curPos, int numBodies)
{
    // Hint, not strictly necessary (compute net acceleration!)
}

__global__ void interact_kernel(float4* newPos, float4* oldPos, float4* newVel, float4* oldVel, float dt, float damping, int numBodies)
{
    // TODO :: update positions based on particle-particle interaction forces!
}



