// -----------------------------------------------------------------------------
//  physics.cu — the simulation kernels
// -----------------------------------------------------------------------------
//  One thread per particle, no shared memory, no synchronisation: the perfect
//  case for the GPU. Every thread reads only its own particle and writes only
//  its own particle, so there is nothing to coordinate.
//
//  The "physics" is deliberately cheap and stable rather than accurate:
//
//    * a softened radial attraction toward (or away from) the cursor,
//      damped by a gaussian so only nearby particles react;
//    * a tangential swirl around the cursor, which is what makes it curl;
//    * a tiny random kick so the cloud never fully freezes;
//    * exponential damping, pulling everything toward rest;
//    * semi-implicit Euler integration, then a soft bounce off the walls.
//
//  Everything is angle-free and uses the fast intrinsics (__sinf, __expf,
//  rsqrtf, __atan2f). The final look came from tuning the constants at the top;
//  do not be shy about changing them.
// -----------------------------------------------------------------------------

#include "particles.cuh"
#include "cuda_check.h"

namespace {

constexpr int kBlock = 256;

constexpr float kTau = 6.28318530718f;
constexpr float kInvTau = 0.15915494309f;  // 1 / (2*pi)

// -----------------------------------------------------------------------------
//  A tiny hash-based random number generator.
//
//  We could use <curand>, but a hash keeps this example dependency-free and
//  shows something useful: given a seed that differs per thread, we get
//  independent-looking streams with no state to carry around.
// -----------------------------------------------------------------------------
__device__ __forceinline__ unsigned wang_hash(unsigned x) {
  x = (x ^ 61u) ^ (x >> 16);
  x *= 9u;
  x = x ^ (x >> 4);
  x *= 0x27d4eb2du;
  x = x ^ (x >> 15);
  return x;
}

__device__ __forceinline__ float rnd01(unsigned& state) {
  // xorshift32 — cheap and more than good enough for visuals.
  state ^= state << 13;
  state ^= state >> 17;
  state ^= state << 5;
  return static_cast<float>(state & 0x00FFFFFFu) * (1.0f / 16777216.0f);
}

// -----------------------------------------------------------------------------
//  An "IQ palette": three cosines with different phases. One line gives you a
//  smooth, saturated rainbow without any branching — a classic creative-coding
//  trick that maps perfectly onto a GPU.
// -----------------------------------------------------------------------------
__device__ __forceinline__ float3 palette(float t, int style) {
  if (style == 0) {  // glacial cyan -> lavender -> molten amber
    const float h = 0.5f + 0.5f * __sinf(kTau * t);
    return make_float3(0.12f + 0.88f * h, 0.38f + 0.40f * (1.0f - h),
                      1.0f - 0.88f * h);
  }
  const float3 phase = style == 1 ? make_float3(0.45f, 0.05f, 0.20f)
                                 : make_float3(0.00f, 0.30f, 0.65f);
  return make_float3(0.52f + 0.48f * __cosf(kTau * (t + phase.x)),
                    0.52f + 0.48f * __cosf(kTau * (t + phase.y)),
                    0.52f + 0.48f * __cosf(kTau * (t + phase.z)));
}

__device__ __forceinline__ unsigned char to_byte(float v) {
  return static_cast<unsigned char>(fminf(fmaxf(v, 0.0f), 1.0f) * 255.0f + 0.5f);
}

// =============================================================================
//  Initialisation
// =============================================================================
__global__ void init_kernel(float4* __restrict__ pos, float4* __restrict__ vel,
                            uchar4* __restrict__ col, int n, unsigned seed,
                            float2 bounds, int scene) {
  const int i = blockIdx.x * kBlock + threadIdx.x;
  if (i >= n) return;

  unsigned s = wang_hash(static_cast<unsigned>(i) * 2654435761u + seed);

  // Uniform point in a disc: sqrt() of a uniform radius is the trick that
  // avoids clustering everything in the middle.
  float angle = rnd01(s) * kTau;
  float r = sqrtf(rnd01(s));
  if (scene == 0) {
    r = (0.12f + r * 0.76f) * fminf(bounds.x, bounds.y);
    angle = (i % 3) * kTau / 3.0f + r * 5.0f + (rnd01(s) - 0.5f) * 0.42f;
  }
  float2 p = make_float2(__cosf(angle) * r * bounds.x * 0.65f,
                               __sinf(angle) * r * bounds.y * 0.65f);
  if (scene == 0) p = make_float2(__cosf(angle) * r, __sinf(angle) * r);
  if (scene == 1) {
    p.x = (rnd01(s) * 2.0f - 1.0f) * bounds.x;
    p.y = (static_cast<float>(i % 5) - 2.0f) * 0.22f +
          0.14f * __sinf(p.x * 2.5f) + (rnd01(s) - 0.5f) * 0.075f;
  }
  if (scene == 2) p = make_float2((rnd01(s) * 2.0f - 1.0f) * bounds.x,
                                 (rnd01(s) * 2.0f - 1.0f) * bounds.y);

  // A tangential kick gives the disc a lazy rotation from frame one.
  const float spin = 0.15f + 0.35f * rnd01(s);
  const float2 v = make_float2(-p.y * spin, p.x * spin);

  pos[i] = make_float4(p.x, p.y, 0.0f, 0.35f + 0.65f * rnd01(s));
  vel[i] = make_float4(v.x, v.y, sqrtf(p.x * p.x + p.y * p.y), rnd01(s));
  const float3 c = palette(r, scene);
  col[i] = make_uchar4(to_byte(c.x), to_byte(c.y), to_byte(c.z), 80);
}

// =============================================================================
//  One simulation step
// =============================================================================
__global__ void update_kernel(float4* __restrict__ pos, float4* __restrict__ vel,
                              uchar4* __restrict__ col, Params p) {
  const int i = blockIdx.x * kBlock + threadIdx.x;
  if (i >= p.count) return;

  float4 P = pos[i];
  float4 V = vel[i];

  // Three cheap analytic fields, not an N-body solver. No neighbour reads.
  // Relax velocity toward a moving field: coherent structures survive even
  // when nobody is touching the mouse. vel.z stores each orbit's birth radius.
  float2 flow;
  if (p.scene == 0) {
    const float r = sqrtf(P.x * P.x + P.y * P.y + 1e-5f);
    const float omega = 0.32f / (0.22f + r);
    const float radial = (V.z - r) * 1.8f;
    flow = make_float2(-P.y * omega + P.x / r * radial,
                       P.x * omega + P.y / r * radial);
  } else if (p.scene == 1) {
    flow = make_float2(0.35f + 0.16f * __cosf(P.y * 4.0f + p.time * 0.3f),
                      0.28f * __cosf(P.x * 2.5f - p.time * 0.45f) +
                      0.12f * __sinf(P.x * 5.0f + P.y * 3.0f));
  } else {
    // Curl of a sinusoidal stream function: divergence-free before damping.
    flow = make_float2(0.42f * __sinf(P.x * 3.0f + p.time * 0.2f) * __cosf(P.y * 3.0f),
                     -0.42f * __cosf(P.x * 3.0f + p.time * 0.2f) * __sinf(P.y * 3.0f));
  }
  const float follow = 1.0f - __expf(-2.0f * p.dt);
  V.x += (flow.x - V.x) * follow;
  V.y += (flow.y - V.y) * follow;

  // ---------------------------------------------------------------------------
  //  Force from the cursor.
  // ---------------------------------------------------------------------------
  const float2 d = make_float2(p.mouse.x - P.x, p.mouse.y - P.y);
  const float r2 = d.x * d.x + d.y * d.y + 2.5e-3f;  // softened: no divide by 0
  const float inv_r = rsqrtf(r2);

  // Gaussian falloff => distant particles ignore the cursor entirely, so the
  // interaction feels like a local disturbance rather than a global pull.
  const float falloff = __expf(-r2 / (p.radius * p.radius));

  // The (1 - core^2/r^2) factor makes the force vanish at a small "core"
  // radius and turn repulsive inside it. Without it a radial attraction would
  // suck every particle into a single dot on the cursor; with it the cloud
  // settles into a vortex instead. Cheap self-limiting behaviour — a very
  // common trick in interactive simulations.
  constexpr float kCore2 = 0.0125f;  // core radius squared (~0.11 world units)
  float force = p.attraction * falloff * (1.0f - kCore2 / r2);
  force = fminf(fmaxf(force, -2500.0f), 2500.0f) * p.dt;

  V.x += d.x * inv_r * force;
  V.y += d.y * inv_r * force;

  // The swirl: same magnitude, rotated 90 degrees. This is what turns a boring
  // radial pull into a vortex you can stir.
  const float2 tangent = make_float2(-d.y, d.x);
  V.x += tangent.x * inv_r * force * p.swirl;
  V.y += tangent.y * inv_r * force * p.swirl;
  V.x += p.mouse_velocity.x * falloff * p.dt * 3.0f;
  V.y += p.mouse_velocity.y * falloff * p.dt * 3.0f;

  // ---------------------------------------------------------------------------
  //  A pinch of noise so the field never goes completely still.
  // ---------------------------------------------------------------------------
  unsigned s = wang_hash(static_cast<unsigned>(i) * 747796405u +
                         static_cast<unsigned>(p.time * 1000.0f));
  V.x += (rnd01(s) - 0.5f) * p.jitter * p.dt;
  V.y += (rnd01(s) - 0.5f) * p.jitter * p.dt;

  // ---------------------------------------------------------------------------
  //  Damping. exp(-k*dt) is the frame-rate-independent way to say
  //  "lose a fixed fraction per second" (naive V *= 0.99 is not).
  // ---------------------------------------------------------------------------
  const float decay = __expf(-p.damping * p.dt);
  V.x *= decay;
  V.y *= decay;
  const float speed2 = V.x * V.x + V.y * V.y;
  if (speed2 > 9.0f) {  // limit impulses from fast cursor jumps / strong clicks
    const float limit = 3.0f * rsqrtf(speed2);
    V.x *= limit; V.y *= limit;
  }

  // ---------------------------------------------------------------------------
  //  Integrate (semi-implicit Euler) and keep everything on screen.
  // ---------------------------------------------------------------------------
  P.x += V.x * p.dt;
  P.y += V.y * p.dt;

  if (p.scene != 0) {
    if (P.x > p.bounds.x) P.x -= 2.0f * p.bounds.x;
    if (P.x < -p.bounds.x) P.x += 2.0f * p.bounds.x;
    if (P.y > p.bounds.y) P.y -= 2.0f * p.bounds.y;
    if (P.y < -p.bounds.y) P.y += 2.0f * p.bounds.y;
  }
  if (P.x > p.bounds.x) { P.x = p.bounds.x;  V.x = -fabsf(V.x) * 0.6f; }
  if (P.x < -p.bounds.x) { P.x = -p.bounds.x; V.x = fabsf(V.x) * 0.6f; }
  if (P.y > p.bounds.y) { P.y = p.bounds.y;  V.y = -fabsf(V.y) * 0.6f; }
  if (P.y < -p.bounds.y) { P.y = -p.bounds.y; V.y = fabsf(V.y) * 0.6f; }

  // ---------------------------------------------------------------------------
  //  Colour: hue follows the *direction* of travel (so neighbours in the flow
  //  share a colour and you see coherent ribbons), brightness follows speed.
  // ---------------------------------------------------------------------------
  const float speed = sqrtf(V.x * V.x + V.y * V.y);
  const float hue = atan2f(V.y, V.x) * kInvTau + 0.5f;
  float3 c = palette(p.scene == 0 ? V.z * 0.9f + V.w * 0.12f : hue * 0.5f + P.x * 0.15f,
                     p.palette);
  const float bright = fminf(1.0f, 0.40f + speed * 0.35f);
  c.x *= bright;
  c.y *= bright;
  c.z *= bright;

  // Pack the speed for the vertex shader; it turns speed into point size.
  P.z = fminf(speed, 3.0f) * 0.5f;

  pos[i] = P;
  vel[i] = V;
  col[i] = make_uchar4(to_byte(c.x), to_byte(c.y), to_byte(c.z),
                      to_byte(0.08f + P.w * 0.12f));
}

}  // namespace

// -----------------------------------------------------------------------------
//  Host-side wrappers. These are just convenience: they work out the launch
//  geometry and keep the kernel names out of the GL code.
// -----------------------------------------------------------------------------
void particles_init(float4* pos, float4* vel, uchar4* col, int count,
                    unsigned seed, float2 bounds, int scene, cudaStream_t stream) {
  const int grid = (count + kBlock - 1) / kBlock;
  init_kernel<<<grid, kBlock, 0, stream>>>(pos, vel, col, count, seed, bounds, scene);
  CUDA_CHECK(cudaGetLastError());
}

void particles_update(float4* pos, float4* vel, uchar4* col, Params p,
                      cudaStream_t stream) {
  const int grid = (p.count + kBlock - 1) / kBlock;
  update_kernel<<<grid, kBlock, 0, stream>>>(pos, vel, col, p);
  CUDA_CHECK(cudaGetLastError());
}
