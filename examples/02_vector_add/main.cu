// -----------------------------------------------------------------------------
//  02 — Vector add: moving data between host and device
// -----------------------------------------------------------------------------
//  Build & run:   make run-02_vector_add
//
//  A kernel cannot see your normal variables. `a`, `b`, `c` on the CPU live in
//  RAM; the GPU has its own memory. So the pattern is always:
//
//        allocate on device      cudaMalloc
//        send data to device     cudaMemcpy(..., cudaMemcpyHostToDevice)
//        launch kernel(s)        kernel<<<grid, block>>>(device_ptrs...)
//        pull data back          cudaMemcpy(..., cudaMemcpyDeviceToHost)
//        free                    cudaFree
//
//  You only ever pass *device pointers* to a kernel. Passing a host pointer is
//  the classic beginner crash.
//
//  Also met here: the bounds check. We launch ceil(n/block) threads which is
//  often not exactly n, so the "extra" threads must do nothing.
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstdio>

#include "cuda_check.h"

// __global__ = runs on GPU, called from CPU.
// __restrict__ promises the compiler the pointers do not alias -> more
// aggressive optimisation, and it is free.
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b, float* __restrict__ c,
                           int n) {
  // Flatten the 1-D grid into one unique index per thread.
  const int i = blockIdx.x * blockDim.x + threadIdx.x;

  // The guard. Without it, the last block writes past the end of the arrays.
  if (i < n) {
    c[i] = a[i] + b[i];
  }
}

int main() {
  const int n = 1 << 20;                      // 1,048,576 floats
  const size_t bytes = n * sizeof(float);

  // ---------------------------------------------------------------------------
  //  1. Host (CPU) side data. Plain std::vector would do; malloc is used so the
  //     symmetry with cudaMalloc is visible.
  // ---------------------------------------------------------------------------
  auto* h_a = static_cast<float*>(std::malloc(bytes));
  auto* h_b = static_cast<float*>(std::malloc(bytes));
  auto* h_c = static_cast<float*>(std::malloc(bytes));  // result from the GPU

  for (int i = 0; i < n; ++i) {
    h_a[i] = static_cast<float>(i);
    h_b[i] = static_cast<float>(i) * 0.5f;
  }

  // ---------------------------------------------------------------------------
  //  2. Device (GPU) side data.
  // ---------------------------------------------------------------------------
  float* d_a = nullptr;
  float* d_b = nullptr;
  float* d_c = nullptr;
  CUDA_CHECK(cudaMalloc(&d_a, bytes));
  CUDA_CHECK(cudaMalloc(&d_b, bytes));
  CUDA_CHECK(cudaMalloc(&d_c, bytes));

  // Only the *inputs* need sending. d_c is write-only, no point copying junk.
  CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

  // ---------------------------------------------------------------------------
  //  3. Launch. 256 threads per block is the usual starting point; the grid
  //     just grows until every element is covered.
  // ---------------------------------------------------------------------------
  const int block = 256;
  const int grid = (n + block - 1) / block;  // ceil(n / block) — integer maths

  std::printf("n = %d, block = %d, grid = %d, total threads = %d\n", n, block,
              grid, block * grid);

  vector_add<<<grid, block>>>(d_a, d_b, d_c, n);
  CUDA_CHECK_KERNEL();

  // ---------------------------------------------------------------------------
  //  4. Bring the result home and check it.
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

  int wrong = 0;
  for (int i = 0; i < n; ++i) {
    const float expected = h_a[i] + h_b[i];
    if (std::fabs(h_c[i] - expected) > 1e-5f) {
      if (wrong++ < 5) {
        std::printf("  mismatch at %d: got %f expected %f\n", i, h_c[i], expected);
      }
    }
  }

  std::printf("%s\n", wrong == 0 ? "OK — every element matches"
                                 : "FAILED — see mismatches above");
  std::printf("c[0]=%g  c[1]=%g  c[%d]=%g\n", h_c[0], h_c[1], n - 1, h_c[n - 1]);

  // ---------------------------------------------------------------------------
  //  5. Always free. cudaFree(device ptr) — never free() a device pointer.
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaFree(d_a));
  CUDA_CHECK(cudaFree(d_b));
  CUDA_CHECK(cudaFree(d_c));
  std::free(h_a);
  std::free(h_b);
  std::free(h_c);
  return EXIT_SUCCESS;
}
