// -----------------------------------------------------------------------------
//  cuda_check.h — the single most useful header while learning CUDA.
// -----------------------------------------------------------------------------
//  Almost every CUDA call returns a cudaError_t that is easy to ignore, and a
//  silently-failing kernel is the #1 source of "why is my output all zeros?".
//  Wrap calls in CUDA_CHECK(...) and you get a file/line + a readable message
//  instead of silence.
//
//  Note the two different things that can go wrong:
//    * CUDA_API_CALL   — the launch / allocation itself failed   -> cudaGetErrorString
//    * KERNEL_EXECUTION— the GPU faulted *after* the launch      -> reported later
//  Kernel launches are asynchronous, so the error often shows up at the *next*
//  call. CUDA_CHECK_KERNEL() forces a sync so nothing escapes.
// -----------------------------------------------------------------------------
#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t err__ = (call);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::fprintf(stderr,                                                     \
                   "\n[CUDA ERROR] %s\n"                                       \
                   "  call: %s\n"                                              \
                   "  at:   %s:%d\n",                                          \
                   cudaGetErrorString(err__), #call, __FILE__, __LINE__);      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// Run after a kernel launch: catches both a bad launch configuration and any
// error raised while the kernel actually ran.
#define CUDA_CHECK_KERNEL()                                                    \
  do {                                                                         \
    CUDA_CHECK(cudaGetLastError());                                            \
    CUDA_CHECK(cudaDeviceSynchronize());                                       \
  } while (0)
