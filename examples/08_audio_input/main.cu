// -----------------------------------------------------------------------------
//  08 — Audio input: live sound in, three smooth control signals out
// -----------------------------------------------------------------------------
//  Build & run:   make run-08_audio_input          (listens to what is playing)
//                 make run-08_audio_input ARGS=--mic
//
//  Lesson 07 turned a block of samples into bass / mid / high levels. Now the
//  samples are real: whatever your computer is playing, or your microphone.
//  The output is what Particle Atlas will use to move particles: three numbers
//  between 0 and 1 that react fast to hits and decay smoothly afterwards.
//
//        sound card ──► 512 new samples ──► sliding 1024 window ──► GPU FFT
//          (PipeWire / PulseAudio)          (pinned host buffer)      │
//                                                                     ▼
//        particles ◄── envelope + auto-gain ◄── 3 band levels ◄── reduction
//          (part 2)        (CPU, 3 floats)       (pinned copy back)
//
//  New things met here:
//    * Real-time thinking. Every 512 samples (10.7 ms at 48 kHz) a new block
//      arrives, and the whole GPU round trip has to finish well before the
//      next one. The program prints how long it takes.
//    * Overlapping windows ("hop size"). Each FFT sees 1024 samples, but we
//      advance by 512, so the levels update twice as often as the window is
//      long.
//    * Pinned memory again (lesson 05): the window and the result live in
//      page-locked host memory, so the copies are fast and can be async.
//    * Envelopes and auto-gain. Raw FFT levels flicker and depend on how loud
//      the track is. A fast attack / slow release follower plus a decaying
//      peak normaliser turns them into signals that look good on screen. This
//      is art direction, not signal processing, just like the particle fields.
//
//  Audio capture uses PulseAudio's "simple" API. On a modern Linux desktop it
//  goes through PipeWire, and it can record the *monitor* of your speakers,
//  i.e. exactly what you hear. Install: `sudo pacman -S libpulse`.
// -----------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cufft.h>
#include <pulse/error.h>
#include <pulse/simple.h>

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

// ---- timing of the analysis -------------------------------------------------
constexpr int   N           = 1024;   // FFT window: 21.3 ms
constexpr int   HOP         = 512;    // new samples per update: 10.7 ms
constexpr int   SAMPLE_RATE = 48000;
constexpr int   BINS        = N / 2 + 1;
constexpr float AMP_SCALE   = 4.0f / N;  // see lesson 07

constexpr int BANDS = 3;
constexpr int BAND_FIRST[BANDS] = {1, 6, 86};          //   47 Hz,  281 Hz, 4 kHz
constexpr int BAND_LAST[BANDS]  = {5, 85, BINS - 1};   //  234 Hz, 3.98 kHz, 24 kHz

// ---- the art-direction knobs ------------------------------------------------
//  Change these, rebuild, and watch the meter. They carry straight over to
//  the particle system.
constexpr float ATTACK_S   = 0.010f;  // how fast a level rises on a hit
constexpr float RELEASE_S  = 0.180f;  // how slowly it falls afterwards
constexpr float PEAK_HOLD  = 4.0f;    // auto-gain forgets old peaks over ~4 s
constexpr float NOISE_GATE = 0.002f;  // below this, treat the band as silence

// -----------------------------------------------------------------------------
//  Kernels (the same three as lesson 07, so each lesson stays self-contained)
// -----------------------------------------------------------------------------
__global__ void apply_hann(const float* __restrict__ in, float* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = in[i] * (0.5f - 0.5f * cospif(2.0f * i / n));
}

__global__ void magnitudes(const cufftComplex* __restrict__ spectrum,
                           float* __restrict__ mag, int bins, float scale) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < bins) {
    const cufftComplex c = spectrum[k];
    mag[k] = sqrtf(c.x * c.x + c.y * c.y) * scale;
  }
}

template <int BLOCK>
__global__ void band_levels(const float* __restrict__ mag, const int* __restrict__ first,
                            const int* __restrict__ last, float* __restrict__ level) {
  __shared__ float partial[BLOCK];
  const int band = blockIdx.x;
  float sum = 0.0f;
  for (int k = first[band] + threadIdx.x; k <= last[band]; k += BLOCK) sum += mag[k] * mag[k];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) level[band] = sqrtf(partial[0]);
}

// -----------------------------------------------------------------------------
//  Turning raw levels into something you would want to animate with
// -----------------------------------------------------------------------------
struct Band {
  float env  = 0.0f;   // smoothed level
  float peak = 0.0f;   // recent loudest value, for auto-gain
};

// One-pole smoothing that does not depend on the update rate: the time
// constant is in seconds, `dt` is how much time this update covers.
static float follow(float current, float target, float tau, float dt) {
  return current + (target - current) * (1.0f - std::exp(-dt / tau));
}

static float shape(Band& b, float raw, float dt) {
  b.env = follow(b.env, raw, raw > b.env ? ATTACK_S : RELEASE_S, dt);
  b.peak = std::fmax(b.env, b.peak * std::exp(-dt / PEAK_HOLD));
  if (b.peak < NOISE_GATE) return 0.0f;
  return b.env / b.peak;  // 0..1, whatever the volume knob says
}

static std::atomic<bool> g_running{true};
static void on_sigint(int) { g_running = false; }

static void meter(const char* label, float v) {
  constexpr int W = 20;
  const int n = static_cast<int>(v * W + 0.5f);
  std::printf("%s ", label);
  for (int i = 0; i < W; ++i) std::printf(i < n ? "█" : "·");
  std::printf("  ");
}

static void usage(const char* argv0) {
  std::printf("usage: %s [--mic] [--device NAME] [--seconds S]\n"
              "  (default)      listen to what is playing (@DEFAULT_MONITOR@)\n"
              "  --mic          listen to the default microphone instead\n"
              "  --device NAME  any PulseAudio/PipeWire source (see `pactl list short sources`)\n"
              "  --seconds S    stop after S seconds (default: run until Ctrl-C)\n",
              argv0);
}

int main(int argc, char** argv) {
  const char* device = "@DEFAULT_MONITOR@";
  float seconds = 0.0f;
  for (int i = 1; i < argc; ++i) {
    if (!std::strcmp(argv[i], "--mic")) device = nullptr;  // nullptr = default source
    else if (!std::strcmp(argv[i], "--device") && i + 1 < argc) device = argv[++i];
    else if (!std::strcmp(argv[i], "--seconds") && i + 1 < argc) seconds = std::atof(argv[++i]);
    else { usage(argv[0]); return EXIT_FAILURE; }
  }

  // ---- 1. open the audio stream ---------------------------------------------
  //  Mono float samples at 48 kHz. `fragsize` asks the server to hand us data
  //  in HOP-sized pieces. Without it PulseAudio buffers ~2 s by default, and
  //  the meter would lag far behind the music.
  const pa_sample_spec spec = {PA_SAMPLE_FLOAT32LE, SAMPLE_RATE, 1};
  pa_buffer_attr attr;
  std::memset(&attr, 0xff, sizeof(attr));  // (uint32_t)-1 = "server default"
  attr.fragsize = HOP * sizeof(float);
  int pa_err = 0;
  pa_simple* audio = pa_simple_new(nullptr, "Particle Atlas", PA_STREAM_RECORD, device,
                                   "lesson 08", &spec, nullptr, &attr, &pa_err);
  if (!audio) {
    std::fprintf(stderr, "could not open audio source '%s': %s\n",
                 device ? device : "(default mic)", pa_strerror(pa_err));
    return EXIT_FAILURE;
  }

  // ---- 2. GPU side, set up once ---------------------------------------------
  float *h_window, *h_level;                         // pinned host memory
  CUDA_CHECK(cudaMallocHost(&h_window, N * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h_level, BANDS * sizeof(float)));
  std::memset(h_window, 0, N * sizeof(float));

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
  CUDA_CHECK(cudaMemcpy(d_first, BAND_FIRST, sizeof(BAND_FIRST), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_last, BAND_LAST, sizeof(BAND_LAST), cudaMemcpyHostToDevice));

  cudaStream_t stream;
  CUDA_CHECK(cudaStreamCreate(&stream));
  cufftHandle plan;
  CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));
  CUFFT_CHECK(cufftSetStream(plan, stream));  // keep the FFT on our stream

  std::signal(SIGINT, on_sigint);
  std::printf("listening to %s  (Ctrl-C to stop)\n"
              "window %d samples, hop %d (%.1f ms per update)\n\n",
              device ? device : "the default microphone", N, HOP,
              1000.0f * HOP / SAMPLE_RATE);

  // ---- 3. the real-time loop ------------------------------------------------
  Band bands[BANDS];
  const float dt = static_cast<float>(HOP) / SAMPLE_RATE;
  const int max_updates = seconds > 0.0f ? static_cast<int>(seconds / dt) : -1;
  double gpu_ms_total = 0.0, gpu_ms_worst = 0.0;
  int updates = 0;

  while (g_running && updates != max_updates) {
    // Slide the window left by HOP and read HOP fresh samples into the end.
    // pa_simple_read blocks until they exist, which paces the whole loop.
    std::memmove(h_window, h_window + HOP, (N - HOP) * sizeof(float));
    if (pa_simple_read(audio, h_window + (N - HOP), HOP * sizeof(float), &pa_err) < 0) {
      std::fprintf(stderr, "\naudio read failed: %s\n", pa_strerror(pa_err));
      break;
    }

    const auto t0 = std::chrono::steady_clock::now();
    CUDA_CHECK(cudaMemcpyAsync(d_samples, h_window, N * sizeof(float),
                               cudaMemcpyHostToDevice, stream));
    apply_hann<<<(N + 255) / 256, 256, 0, stream>>>(d_samples, d_windowed, N);
    CUFFT_CHECK(cufftExecR2C(plan, d_windowed, d_spectrum));
    magnitudes<<<(BINS + 255) / 256, 256, 0, stream>>>(d_spectrum, d_mag, BINS, AMP_SCALE);
    band_levels<128><<<BANDS, 128, 0, stream>>>(d_mag, d_first, d_last, d_level);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_level, d_level, BANDS * sizeof(float),
                               cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    const double ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (updates > 0) {  // the first round trip includes one-off warm-up costs
      gpu_ms_total += ms;
      gpu_ms_worst = std::fmax(gpu_ms_worst, ms);
    }
    ++updates;

    float v[BANDS];
    for (int b = 0; b < BANDS; ++b) v[b] = shape(bands[b], h_level[b], dt);

    // Redraw the meter in place a few times a second; printing every 10 ms
    // would cost more than the GPU work.
    if (updates % 3 == 0) {
      std::printf("\r");
      meter("bass", v[0]);
      meter("mid", v[1]);
      meter("high", v[2]);
      const bool silent = bands[0].peak < NOISE_GATE && bands[1].peak < NOISE_GATE &&
                          bands[2].peak < NOISE_GATE;
      std::printf("%s", silent ? "(silence: play something)" : "                         ");
      std::fflush(stdout);
    }
  }

  if (updates > 1) {
    std::printf("\n\n%d updates. GPU round trip: %.3f ms average, %.3f ms worst "
                "(budget %.1f ms)\n",
                updates, gpu_ms_total / (updates - 1), gpu_ms_worst, 1000.0f * dt);
  }

  // Try this:
  //   * Set RELEASE_S to 0.02: the meter turns twitchy. Set it to 1.0: it
  //     floats. Somewhere in between is what music "looks" like to you.
  //   * Make HOP 256: twice as many updates, same window. Does it feel tighter?
  //   * Remove the auto-gain (return b.env * 20.0f): now the volume knob drives
  //     the visuals, and quiet tracks barely move.

  CUFFT_CHECK(cufftDestroy(plan));
  CUDA_CHECK(cudaStreamDestroy(stream));
  CUDA_CHECK(cudaFree(d_samples));
  CUDA_CHECK(cudaFree(d_windowed));
  CUDA_CHECK(cudaFree(d_spectrum));
  CUDA_CHECK(cudaFree(d_mag));
  CUDA_CHECK(cudaFree(d_level));
  CUDA_CHECK(cudaFree(d_first));
  CUDA_CHECK(cudaFree(d_last));
  CUDA_CHECK(cudaFreeHost(h_window));
  CUDA_CHECK(cudaFreeHost(h_level));
  pa_simple_free(audio);
  return EXIT_SUCCESS;
}
