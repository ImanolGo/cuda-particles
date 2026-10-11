// GPU-only regression checks: no display or OpenGL context required.
#include <cmath>
#include <cstdio>
#include <vector>
#include "cuda_check.h"
#include "particles.cuh"

int main() {
  bool correct = true;
  for (int n : {1, 257, 1027, 100003}) {
    float4 *pos = nullptr, *vel = nullptr;
    uchar4* col = nullptr;
    const size_t bytes = static_cast<size_t>(n + 1) * sizeof(float4);
    CUDA_CHECK(cudaMalloc(&pos, bytes));
    CUDA_CHECK(cudaMalloc(&vel, bytes));
    CUDA_CHECK(cudaMalloc(&col, static_cast<size_t>(n + 1) * sizeof(uchar4)));
    for (int scene = 0; scene < 3; ++scene) {
      CUDA_CHECK(cudaMemset(pos, 0, bytes));
      CUDA_CHECK(cudaMemset(vel, 0, bytes));
      CUDA_CHECK(cudaMemset(col, 0, static_cast<size_t>(n + 1) * sizeof(uchar4)));
      Params p{};
      p.bounds = make_float2(1.777f, 1);
      p.dt = 1.0f / 120;
      p.attraction = 12;
      p.swirl = 0.85f;
      p.damping = 0.12f;
      p.jitter = 0.15f;
      p.radius = 0.3f;
      p.count = n;
      p.scene = p.palette = scene;
      particles_init(pos, vel, col, n, 1, p.bounds, scene, 0);
      for (int step = 0; step < 240; ++step) {
        p.time = step * p.dt;
        p.mouse = make_float2(0.6f * std::cos(p.time), 0.4f * std::sin(p.time));
        p.attraction = step < 120 ? 12.0f : -12.0f;
        particles_update(pos, vel, col, p, 0);
      }
      std::vector<float4> positions(n + 1), velocities(n + 1);
      std::vector<uchar4> colours(n + 1);
      CUDA_CHECK(cudaMemcpy(positions.data(), pos, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(velocities.data(), vel, bytes, cudaMemcpyDeviceToHost));
      CUDA_CHECK(cudaMemcpy(colours.data(), col, colours.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
      for (int i = 0; i < n; ++i) {
        const auto P = positions[i], V = velocities[i];
        correct &= std::isfinite(P.x) && std::isfinite(P.y) && std::isfinite(P.z) && std::isfinite(P.w);
        correct &= std::isfinite(V.x) && std::isfinite(V.y) && std::isfinite(V.z) && std::isfinite(V.w);
        correct &= std::fabs(P.x) <= p.bounds.x && std::fabs(P.y) <= p.bounds.y;
        correct &= V.x * V.x + V.y * V.y <= 9.0001f && colours[i].w > 0;
      }
      // One extra element catches writes by excess threads in the final block.
      const auto P = positions[n], V = velocities[n];
      correct &= P.x == 0 && P.y == 0 && P.z == 0 && P.w == 0;
      correct &= V.x == 0 && V.y == 0 && V.z == 0 && V.w == 0;
      const auto C = colours[n];
      correct &= C.x == 0 && C.y == 0 && C.z == 0 && C.w == 0;
    }
    CUDA_CHECK(cudaFree(pos)); CUDA_CHECK(cudaFree(vel)); CUDA_CHECK(cudaFree(col));
  }
  std::printf("physics: finite, bounded, speed-limited, tail guards intact: %s\n", correct ? "yes" : "NO");
  return correct ? EXIT_SUCCESS : EXIT_FAILURE;
}
