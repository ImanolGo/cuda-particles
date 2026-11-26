// -----------------------------------------------------------------------------
//  audio.cu — the lesson 07/08 pipeline, wired for the particle system
// -----------------------------------------------------------------------------
//  window (pinned) -> Hann -> cuFFT R2C -> magnitudes -> 3 band levels -> shape
//
//  Two details matter once this runs next to a renderer:
//
//    * The analysis gets its own *non-blocking* stream. The particle kernels
//      run on the legacy default stream, which implicitly waits for every
//      ordinary stream; a non-blocking stream opts out of that, so audio work
//      never stalls a frame and a frame never stalls the audio.
//    * Live capture runs on a separate thread because pa_simple_read blocks
//      until the next 10.7 ms of sound exists. The render loop just reads the
//      latest three numbers under a mutex.
// -----------------------------------------------------------------------------
#include "audio.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <mutex>
#include <thread>
#include <vector>

#include <cufft.h>

#include "cuda_check.h"

#ifdef PARTICLES_AUDIO
#include <pulse/error.h>
#include <pulse/simple.h>
#endif

namespace {

constexpr int N = 1024;            // FFT window
[[maybe_unused]] constexpr int HOP = 512;  // live update step (live capture only)
constexpr int BINS = N / 2 + 1;
constexpr int BANDS = 3;
constexpr float AMP_SCALE = 4.0f / N;
constexpr float LIVE_RATE = 48000.0f;

// Band edges in Hz, turned into FFT bins once the sample rate is known.
constexpr float BAND_HZ[BANDS + 1] = {40.0f, 250.0f, 4000.0f, 16000.0f};

// Envelope and auto-gain, as in lesson 08.
constexpr float ATTACK_S = 0.010f;
constexpr float RELEASE_S = 0.180f;
constexpr float PEAK_HOLD = 4.0f;
constexpr float NOISE_GATE = 0.002f;

#define CUFFT_CHECK(call)                                                      \
  do {                                                                         \
    cufftResult res__ = (call);                                                \
    if (res__ != CUFFT_SUCCESS) {                                              \
      std::fprintf(stderr, "[cuFFT ERROR] code %d at %s:%d\n",                 \
                   static_cast<int>(res__), __FILE__, __LINE__);               \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

__global__ void apply_hann(const float* __restrict__ in, float* __restrict__ out) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) out[i] = in[i] * (0.5f - 0.5f * cospif(2.0f * i / N));
}

__global__ void magnitudes(const cufftComplex* __restrict__ spectrum, float* __restrict__ mag) {
  const int k = blockIdx.x * blockDim.x + threadIdx.x;
  if (k < BINS) {
    const cufftComplex c = spectrum[k];
    mag[k] = sqrtf(c.x * c.x + c.y * c.y) * AMP_SCALE;
  }
}

struct BandRange { int first[BANDS], last[BANDS]; };

template <int BLOCK>
__global__ void band_levels(const float* __restrict__ mag, BandRange r, float* __restrict__ level) {
  __shared__ float partial[BLOCK];
  const int band = blockIdx.x;
  float sum = 0.0f;
  for (int k = r.first[band] + threadIdx.x; k <= r.last[band]; k += BLOCK) sum += mag[k] * mag[k];
  partial[threadIdx.x] = sum;
  __syncthreads();
  for (int stride = BLOCK / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) partial[threadIdx.x] += partial[threadIdx.x + stride];
    __syncthreads();
  }
  if (threadIdx.x == 0) level[band] = sqrtf(partial[0]);
}

// -----------------------------------------------------------------------------
//  The GPU half: owns the plan, the stream and the buffers.
// -----------------------------------------------------------------------------
struct Analyzer {
  cufftHandle plan = 0;
  cudaStream_t stream = nullptr;
  float* h_window = nullptr;   // pinned: callers fill this, then call run()
  float* h_level = nullptr;    // pinned
  float *d_samples = nullptr, *d_windowed = nullptr, *d_mag = nullptr, *d_level = nullptr;
  cufftComplex* d_spectrum = nullptr;
  BandRange bands{};

  void init(float sample_rate) {
    const float bin_hz = sample_rate / N;
    for (int b = 0; b < BANDS; ++b) {
      bands.first[b] = std::clamp(static_cast<int>(std::ceil(BAND_HZ[b] / bin_hz)), 1, BINS - 1);
      bands.last[b] = std::clamp(static_cast<int>(std::floor(BAND_HZ[b + 1] / bin_hz)) - 1,
                                 bands.first[b], BINS - 1);
    }
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    CUDA_CHECK(cudaMallocHost(&h_window, N * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(&h_level, BANDS * sizeof(float)));
    std::memset(h_window, 0, N * sizeof(float));
    CUDA_CHECK(cudaMalloc(&d_samples, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_windowed, N * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_spectrum, BINS * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_mag, BINS * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_level, BANDS * sizeof(float)));
    CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));
    CUFFT_CHECK(cufftSetStream(plan, stream));
  }

  void run(float out[BANDS]) {
    CUDA_CHECK(cudaMemcpyAsync(d_samples, h_window, N * sizeof(float), cudaMemcpyHostToDevice, stream));
    apply_hann<<<(N + 255) / 256, 256, 0, stream>>>(d_samples, d_windowed);
    CUFFT_CHECK(cufftExecR2C(plan, d_windowed, d_spectrum));
    magnitudes<<<(BINS + 255) / 256, 256, 0, stream>>>(d_spectrum, d_mag);
    band_levels<128><<<BANDS, 128, 0, stream>>>(d_mag, bands, d_level);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpyAsync(h_level, d_level, BANDS * sizeof(float), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    for (int b = 0; b < BANDS; ++b) out[b] = h_level[b];
  }

  void destroy() {
    if (plan) cufftDestroy(plan);
    cudaFree(d_samples); cudaFree(d_windowed); cudaFree(d_spectrum);
    cudaFree(d_mag); cudaFree(d_level);
    cudaFreeHost(h_window); cudaFreeHost(h_level);
    if (stream) cudaStreamDestroy(stream);
  }
};

// The CPU half: raw levels -> 0..1 signals that look good on screen.
struct Shaper {
  float env[BANDS] = {};
  float peak[BANDS] = {};

  AudioLevels step(const float raw[BANDS], float dt) {
    float v[BANDS];
    for (int b = 0; b < BANDS; ++b) {
      const float tau = raw[b] > env[b] ? ATTACK_S : RELEASE_S;
      if (dt > 0) env[b] += (raw[b] - env[b]) * (1.0f - std::exp(-dt / tau));
      peak[b] = std::fmax(env[b], peak[b] * std::exp(-dt / PEAK_HOLD));
      v[b] = peak[b] < NOISE_GATE ? 0.0f : env[b] / peak[b];
    }
    return {v[0], v[1], v[2]};
  }
};

// -----------------------------------------------------------------------------
//  WAV reader: PCM 16/24-bit or float32, any channel count, mixed to mono.
// -----------------------------------------------------------------------------
bool load_wav(const char* path, std::vector<float>& mono, float& rate, char* error, size_t size) {
  std::ifstream file(path, std::ios::binary);
  if (!file) { std::snprintf(error, size, "cannot open %s", path); return false; }
  std::vector<unsigned char> d((std::istreambuf_iterator<char>(file)), {});
  auto u16 = [&](size_t o) { return static_cast<unsigned>(d[o] | d[o + 1] << 8); };
  auto u32 = [&](size_t o) { return u16(o) | u16(o + 2) << 16; };
  if (d.size() < 12 || std::memcmp(d.data(), "RIFF", 4) || std::memcmp(d.data() + 8, "WAVE", 4)) {
    std::snprintf(error, size, "%s is not a WAV file (convert with: ffmpeg -i in.mp3 out.wav)", path);
    return false;
  }
  unsigned format = 0, channels = 0, bits = 0;
  size_t data = 0, data_size = 0;
  for (size_t o = 12; o + 8 <= d.size();) {
    const size_t len = u32(o + 4);
    if (!std::memcmp(d.data() + o, "fmt ", 4) && o + 24 <= d.size()) {
      format = u16(o + 8); channels = u16(o + 10); rate = static_cast<float>(u32(o + 12));
      bits = u16(o + 22);
      if (format == 0xFFFE && o + 34 <= d.size()) format = u16(o + 32);  // WAVE_FORMAT_EXTENSIBLE
    } else if (!std::memcmp(d.data() + o, "data", 4)) {
      data = o + 8; data_size = std::min(len, d.size() - data);
    }
    o += 8 + len + (len & 1);
  }
  const bool pcm = format == 1 && (bits == 16 || bits == 24);
  const bool flt = format == 3 && bits == 32;
  if (!data || !channels || rate <= 0 || (!pcm && !flt)) {
    std::snprintf(error, size, "%s: unsupported WAV (need 16/24-bit PCM or float32)", path);
    return false;
  }
  const size_t bytes = bits / 8, frames = data_size / (bytes * channels);
  mono.assign(frames, 0.0f);
  for (size_t f = 0; f < frames; ++f) {
    float sum = 0.0f;
    for (unsigned c = 0; c < channels; ++c) {
      const unsigned char* s = d.data() + data + (f * channels + c) * bytes;
      if (flt) { float v; std::memcpy(&v, s, 4); sum += v; }
      else if (bits == 16) sum += static_cast<int16_t>(s[0] | s[1] << 8) / 32768.0f;
      else sum += static_cast<int32_t>((s[0] << 8) | (s[1] << 16) | (s[2] << 24)) / 2147483648.0f;
    }
    mono[f] = sum / channels;
  }
  return true;
}

}  // namespace

// -----------------------------------------------------------------------------
struct AudioInput {
  bool live = false;
  Analyzer analyzer;
  Shaper shaper;

  // file source
  std::vector<float> samples;
  float rate = LIVE_RATE;

  // live source
  std::thread worker;
  std::atomic<bool> running{false};
  std::mutex mutex;
  AudioLevels latest;
#ifdef PARTICLES_AUDIO
  pa_simple* pa = nullptr;
#endif
};

bool audio_live_supported() {
#ifdef PARTICLES_AUDIO
  return true;
#else
  return false;
#endif
}

AudioInput* audio_open_file(const char* path, char* error, size_t size) {
  auto* a = new AudioInput;
  if (!load_wav(path, a->samples, a->rate, error, size)) { delete a; return nullptr; }
  a->analyzer.init(a->rate);
  std::printf("audio       : %s (%.1f s at %.0f Hz)\n", path, a->samples.size() / a->rate, a->rate);
  return a;
}

AudioInput* audio_open_live(const char* device, char* error, size_t size) {
#ifdef PARTICLES_AUDIO
  const pa_sample_spec spec = {PA_SAMPLE_FLOAT32LE, static_cast<uint32_t>(LIVE_RATE), 1};
  pa_buffer_attr attr;
  std::memset(&attr, 0xff, sizeof attr);
  attr.fragsize = HOP * sizeof(float);   // small fragments = low latency
  int err = 0;
  pa_simple* pa = pa_simple_new(nullptr, "Particle Atlas", PA_STREAM_RECORD, device,
                                "particles", &spec, nullptr, &attr, &err);
  if (!pa) {
    std::snprintf(error, size, "cannot open audio source '%s': %s",
                  device ? device : "default microphone", pa_strerror(err));
    return nullptr;
  }
  auto* a = new AudioInput;
  a->live = true;
  a->pa = pa;
  a->analyzer.init(LIVE_RATE);
  int cuda_device = 0;
  CUDA_CHECK(cudaGetDevice(&cuda_device));
  a->running = true;
  a->worker = std::thread([a, cuda_device] {
    CUDA_CHECK(cudaSetDevice(cuda_device));   // the CUDA device is per thread
    float* w = a->analyzer.h_window;
    int err = 0;
    while (a->running) {
      std::memmove(w, w + HOP, (N - HOP) * sizeof(float));
      if (pa_simple_read(a->pa, w + (N - HOP), HOP * sizeof(float), &err) < 0) {
        std::fprintf(stderr, "audio read failed: %s\n", pa_strerror(err));
        break;
      }
      float raw[BANDS];
      a->analyzer.run(raw);
      const AudioLevels v = a->shaper.step(raw, HOP / LIVE_RATE);
      std::lock_guard<std::mutex> lock(a->mutex);
      a->latest = v;
    }
  });
  std::printf("audio       : listening to %s\n", device ? device : "the default microphone");
  return a;
#else
  (void)device;
  std::snprintf(error, size,
                "this build has no live audio: install libpulse and rebuild "
                "(--audio-file works without it)");
  return nullptr;
#endif
}

AudioLevels audio_levels(AudioInput* a, double time, float dt) {
  if (!a) return {};
  if (a->live) {
    std::lock_guard<std::mutex> lock(a->mutex);
    return a->latest;
  }
  // The window that *ends* at the current time: what you would just have heard.
  const long end = static_cast<long>(time * a->rate);
  const long total = static_cast<long>(a->samples.size());
  for (int i = 0; i < N; ++i) {
    const long s = end - N + i;
    a->analyzer.h_window[i] = (s >= 0 && s < total) ? a->samples[s] : 0.0f;
  }
  float raw[BANDS];
  a->analyzer.run(raw);
  return a->shaper.step(raw, dt);
}

void audio_close(AudioInput* a) {
  if (!a) return;
  if (a->live) {
    a->running = false;   // the worker notices within one 10.7 ms read
    if (a->worker.joinable()) a->worker.join();
#ifdef PARTICLES_AUDIO
    pa_simple_free(a->pa);
#endif
  }
  a->analyzer.destroy();
  delete a;
}
