// -----------------------------------------------------------------------------
//  audio.h — music in, three control signals out (lessons 07 and 08, packaged)
// -----------------------------------------------------------------------------
//  Two sources, one result:
//
//    live  (--audio)       a capture thread reads the sound card, runs the GPU
//                          FFT on its own stream, and publishes the latest
//                          levels. The render loop only ever reads three floats.
//    file  (--audio-file)  a WAV analysed at the simulation clock. Frame N
//                          always hears the same 21 ms of music, so recordings
//                          made with --frames are reproducible.
//
//  Each level is 0..1 after envelope following and auto-gain (see lesson 08).
// -----------------------------------------------------------------------------
#pragma once

#include <cstddef>

struct AudioLevels {
  float bass = 0.0f;
  float mid = 0.0f;
  float high = 0.0f;
};

struct AudioInput;  // opaque: callers never see the threads or CUDA buffers

// True when this build can capture live audio (libpulse was found).
bool audio_live_supported();

// device: a PulseAudio/PipeWire source name, "@DEFAULT_MONITOR@" for what is
// playing, or nullptr for the default microphone. Returns nullptr on failure
// and writes the reason into `error`.
AudioInput* audio_open_live(const char* device, char* error, size_t error_size);
AudioInput* audio_open_file(const char* path, char* error, size_t error_size);

// Live: the newest levels (time and dt are ignored).
// File: analyse the window ending at `time` seconds, advancing envelopes by dt.
AudioLevels audio_levels(AudioInput* audio, double time, float dt);

void audio_close(AudioInput* audio);
