// -----------------------------------------------------------------------------
//  particles.cuh — the shared vocabulary between host and device
// -----------------------------------------------------------------------------
//  Keeping this in one small header means the GL host code and the CUDA kernels
//  agree on the data layout without any copying or conversion.
#pragma once

#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
//  A particle lives in three parallel arrays (a "structure of arrays", which
//  is what the GPU likes best):
//
//      pos : x, y  = world position      (orthographic space, y up)
//            z     = speed, packed here so the vertex shader can size the point
//            w     = reserved
//      vel : x, y  = velocity
//      col : rgba  = 8-bit colour, already normalised for the vertex shader
//
//  pos and col live inside OpenGL vertex buffers, so there is no copy: CUDA
//  writes, OpenGL reads. vel is private to CUDA and lives in plain VRAM.
// -----------------------------------------------------------------------------

// Everything the simulation needs to know about "right now".
// Note the order: float2 members come first so the struct stays 4-byte aligned
// on both sides.
struct Params {
  float2 mouse;       // cursor position, in world space
  float2 bounds;      // half-extents of the visible area
  float dt;           // seconds since the previous frame
  float time;         // seconds since start (used to reseed the jitter)
  float attraction;   // > 0 pull toward the cursor, < 0 push away, 0 idle
  float swirl;        // how much the cursor twists the field (0 = none)
  float damping;      // velocity decay, in units of 1/second
  float jitter;       // random agitation (keeps the cloud breathing)
  float radius;       // gaussian falloff radius of the cursor force
  int count;          // number of particles
};

// Fill the buffers with a gently rotating disc of particles.
void particles_init(float4* pos, float4* vel, uchar4* col, int count,
                    unsigned seed, float2 bounds, cudaStream_t stream);

// Advance the simulation by one step.
void particles_update(float4* pos, float4* vel, uchar4* col, Params p,
                      cudaStream_t stream);
