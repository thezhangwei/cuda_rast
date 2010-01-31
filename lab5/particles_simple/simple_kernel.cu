
__global__ void simple_kernel(float4* pos, float4* vels, unsigned int width, unsigned int height, float time, float rand, float prand)
{
    // Indices into the VBO data. Roughly like texture coordinates from GLSL.
    unsigned int x = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int y = blockIdx.y*blockDim.y + threadIdx.y;

    // TODO :: Write some value to pos[y*width + x], and make a particle system!
}
