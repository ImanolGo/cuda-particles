// -----------------------------------------------------------------------------
//  07 — Audio FFT: from samples to "how much bass is there right now?"
// -----------------------------------------------------------------------------
//  Build & run:   make run-07_audio_fft
//
//  Part 2 of Particle Atlas makes the particles listen. Before any microphone
//  is involved, this lesson answers the core question on a signal we fully
//  control: given a block of audio samples, how loud are the bass, mids and
//  highs?
//
//  The pipeline is the one a music visualiser runs every frame:
//
//        samples ──► window ──► FFT ──► magnitudes ──► 3 band levels
//        (1024)     (kernel)   (cuFFT)   (kernel)      (kernel + shared mem)
//
//  New things met here:
//    * cuFFT — NVIDIA's FFT library. You make a *plan* once (it picks the
//      algorithm for your size and GPU), then execute it as often as you like.
//      Its calls return cufftResult, not cudaError_t, so they need their own
//      check macro.
//    * Real-to-complex (R2C) transforms: N real samples give only N/2 + 1
//      complex bins, because the other half is a mirror image.
//    * Windowing: chopping audio into blocks creates fake frequencies at the
//      cut edges ("spectral leakage"). Fading each block in and out with a
//      Hann window keeps the spectrum clean.
//    * A shared-memory reduction (lesson 04, again) to sum each band.
//
//  The signal is three pure tones: a bass note, a mid tone and a hiss-like high
//  tone. A slow CPU DFT checks the GPU answer, so the program fails loudly if
//  anything is off.
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cufft.h>

#include "cuda_check.h"

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    cufftResult res__ = (call);                                                \
    if (res__ != CUFFT_SUCCESS) {                                              \
      std::fprintf(stderr, "\n[cuFFT ERROR] code %d\n  call: %s\n  at:   %s:%d\n", \
                   static_cast<int>(res__), #call, __FILE__, __LINE__);        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// ---- the audio block --------------------------------------------------------
constexpr int   N           = 1024;      // samples per block: ~21 ms at 48 kHz
constexpr float SAMPLE_RATE = 48000.0f;
constexpr int   BINS        = N / 2 + 1; // R2C output size
constexpr float BIN_HZ      = SAMPLE_RATE / N;  // 46.875 Hz per bin

// Tones placed exactly on bin centres, so the expected answer is easy to state.
struct Tone { int bin; float amplitude; const char* name; };
constexpr Tone TONES[3] = {
    {  2, 1.00f, "bass"},   //   93.75 Hz
    { 32, 0.50f, "mid "},   // 1500.00 Hz
    {160, 0.25f, "high"},   // 7500.00 Hz
};

// Band edges in bins. Bin 0 is DC (the average level), not sound, so skip it.
//   bass:   47 –  234 Hz     mid:  281 – 3984 Hz     high: 4031 – 24000 Hz
constexpr int BANDS = 3;
constexpr int BAND_FIRST[BANDS] = {1, 6, 86};
constexpr int BAND_LAST[BANDS]  = {5, 85, BINS - 1};  // inclusive

// The Hann window halves the signal on average (coherent gain 0.5), and a
// one-sided spectrum needs x2. Scaling by 4/N makes a sine of amplitude A
// show up as A in its bin.
constexpr float AMP_SCALE = 4.0f / N;

// -----------------------------------------------------------------------------
//  Kernels
// -----------------------------------------------------------------------------

// One thread per sample: fade the block in and out.
__global__ void apply_hann(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    const float w = 0.5f - 0.5f * cospif(2.0f * i / n);  // cospif(x) = cos(pi*x)
    out[i] = in[i] * w;
  }
}

// One thread per frequency bin: complex number -> amplitude.
__global__ void magnitudes(const cufftComplex* __restrict__ spectrum,
                           float* __restrict__ mag, int bins, float scale) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < bins) {
    const cufftComplex c = spectrum[k];
    mag[k] = sqrtf(c.x * c.x + c.y * c.y) * scale;
  }
}

// One block per band. Each thread sums the energy (mag^2) of a few bins with a
// block-stride loop, then the block folds those partial sums together in shared
// memory. The band level is sqrt(total energy): the loudness of everything in
// that band combined.
template <int BLOCK>
__global__ void band_levels(const float* __restrict__ mag,
                            const int* __restrict__ first,
                            const int* __restrict__ last,
                            float* __restrict__ level) {
  __shared__ float partial[BLOCK];
  const int band = blockIdx.x;

  float sum = 0.0f;
  for (int k = first[band] + threadIdx.x; k <= last[band]; k += BLOCK)
    sum += mag[k] * mag[k];
  partial[threadIdx.x] = sum;
  __syncthreads();

  // Tree reduction: 128 -> 64 -> ... -> 1. Same shape as lesson 04.
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) level[band] = sqrtf(partial[0]);
}

// -----------------------------------------------------------------------------
//  CPU reference: a direct O(N^2) DFT. Slow, obvious, and easy to trust.
// -----------------------------------------------------------------------------
static std::vector<float> cpu_magnitudes(const std::vector<float>& x) {
  std::vector<float> mag(BINS);
  for (int k = 0; k < BINS; ++k) {
    double re = 0.0, im = 0.0;
    for (int i = 0; i < N; ++i) {
      const double w = 0.5 - 0.5 * std::cos(2.0 * M_PI * i / N);
      const double a = -2.0 * M_PI * k * i / N;
      re += x[i] * w * std::cos(a);
      im += x[i] * w * std::sin(a);
    }
    mag[k] = static_cast<float>(std::sqrt(re * re + im * im) * AMP_SCALE);
  }
  return mag;
}

static void bar(const char* label, float value, float full_scale) {
  const int width = static_cast<int>(40.0f * value / full_scale + 0.5f);
  std::printf("  %s %6.3f  ", label, value);
  for (int i = 0; i < width; ++i) std::printf("█");
  std::printf("\n");
}

int main() {
  // ---- 1. a synthetic block of audio ----------------------------------------
  std::vector<float> h_samples(N);
  for (int i = 0; i < N; ++i) {
    float s = 0.0f;
    for (const Tone& t : TONES)
      s += t.amplitude * std::sin(2.0f * static_cast<float>(M_PI) * t.bin * i / N);
    h_samples[i] = s;
  }
  std::printf("block: %d samples @ %.0f Hz  (%.1f ms), %d bins of %.3f Hz\n\n",
              N, SAMPLE_RATE, 1000.0f * N / SAMPLE_RATE, BINS, BIN_HZ);

  // ---- 2. device buffers ----------------------------------------------------
  float *d_samples, *d_windowed, *d_mag, *d_level;
  int *d_first, *d_last;
  cufftComplex* d_spectrum;
  CUDA_CHECK(cudaMalloc(&d_samples, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_windowed, N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_spectrum, BINS * sizeof(cufftComplex)));
  CUDA_CHECK(cudaMalloc(&d_mag, BINS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_level, BANDS * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_first, sizeof(BAND_FIRST)));
  CUDA_CHECK(cudaMalloc(&d_last, sizeof(BAND_LAST)));
  CUDA_CHECK(cudaMemcpy(d_samples, h_samples.data(), N * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_first, BAND_FIRST, sizeof(BAND_FIRST), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_last, BAND_LAST, sizeof(BAND_LAST), cudaMemcpyHostToDevice));

  // ---- 3. plan once, execute every frame ------------------------------------
  //  Planning can allocate memory and benchmark algorithms, so it is slow. In
  //  the particle system the plan is made at startup and reused 60x a second.
  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));

  const int block = 256;
  apply_hann<<<(N + block - 1) / block, block>>>(d_samples, d_windowed, N);
  CUDA_CHECK(cudaGetLastError());
  CUFFT_CHECK(cufftExecR2C(plan, d_windowed, d_spectrum));
  magnitudes<<<(BINS + block - 1) / block, block>>>(d_spectrum, d_mag, BINS, AMP_SCALE);
  CUDA_CHECK(cudaGetLastError());
  band_levels<128><<<BANDS, 128>>>(d_mag, d_first, d_last, d_level);
  CUDA_CHECK_KERNEL();

  std::vector<float> h_mag(BINS);
  float h_level[BANDS];
  CUDA_CHECK(cudaMemcpy(h_mag.data(), d_mag, BINS * sizeof(float), cudaMemcpyDeviceToHost));
  CUDA_CHECK(cudaMemcpy(h_level, d_level, sizeof(h_level), cudaMemcpyDeviceToHost));

  // ---- 4. what the particles will "hear" ------------------------------------
  std::printf("tone peaks (expected amplitude -> measured):\n");
  int wrong = 0;
  for (const Tone& t : TONES) {
    const float got = h_mag[t.bin];
    const bool ok = std::fabs(got - t.amplitude) < 1e-3f;
    wrong += !ok;
    std::printf("  %s %8.2f Hz   %.3f -> %.3f  %s\n", t.name, t.bin * BIN_HZ,
                t.amplitude, got, ok ? "ok" : "WRONG");
  }

  const char* band_names[BANDS] = {"bass", "mid ", "high"};
  std::printf("\nband levels:\n");
  for (int b = 0; b < BANDS; ++b) bar(band_names[b], h_level[b], h_level[0]);

  // ---- 5. check against the CPU ---------------------------------------------
  const std::vector<float> ref = cpu_magnitudes(h_samples);
  float max_err = 0.0f;
  for (int k = 0; k < BINS; ++k) max_err = std::fmax(max_err, std::fabs(ref[k] - h_mag[k]));
  for (int b = 0; b < BANDS; ++b) {
    double e = 0.0;
    for (int k = BAND_FIRST[b]; k <= BAND_LAST[b]; ++k) e += double(ref[k]) * ref[k];
    if (std::fabs(std::sqrt(e) - h_level[b]) > 1e-3) ++wrong;
  }
  if (max_err > 1e-3f) ++wrong;
  std::printf("\nmax |GPU - CPU DFT| over %d bins: %.2e  | agree: %s\n", BINS,
              max_err, wrong == 0 ? "yes" : "NO");

  // Try this:
  //   * Remove the window (copy d_samples straight into the FFT) and move a
  //     tone off its bin centre, e.g. bin 32.5. Watch energy leak into
  //     every band.
  //   * Change N to 4096: finer frequency detail, but each block now covers
  //     85 ms, so the visuals would react later. That trade-off is the whole
  //     game in audio-reactive work.

  CUFFT_CHECK(cufftDestroy(plan));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_windowed));
  CUDA_CHECK(cudaFree(d_spectrum));
  CUDA_CHECK(cudaFree(d_mag));
  CUDA_CHECK(cudaFree(d_level));
  CUDA_CHECK(cudaFree(d_first));
  CUDA_CHECK(cudaFree(d_last));
  return wrong == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
