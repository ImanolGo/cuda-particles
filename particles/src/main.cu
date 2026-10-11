// =============================================================================
//  CUDA Particles — the final project
// =============================================================================
//  Build & run:   make run-particles
//                 make run-particles ARGS="--count 1000000 --fade 0.05"
//
//  A GPU-resident particle system you stir with the mouse. The simulation
//  lives entirely in CUDA; the pixels go straight to the screen through
//  OpenGL. In the default interop path no particle data is copied by the CPU — CUDA
//  writes into OpenGL's own vertex buffers via the interop you met in
//  example 06.
//
//  Everything you learned on the way here is used:
//
//    01 hello              kernel launches, thread/block indices
//    02 vector_add         cudaMalloc / memcpy, error checking
//    03 index_math         one thread per element, launch geometry
//    05 timing_streams     async work and the distinction between launch/execution
//    06 interop            cudaGraphicsMap / Unmap, shared VBOs
//
//  New here: analytic flow fields, persistent HDR trails, and bloom/tone mapping.
//  --staged is an explicit, slower copy-based path for non-NVIDIA GL contexts.
//
//  Controls
//  --------
//     move the mouse    the cloud follows and swirls around the cursor
//     left button       strong attraction
//     right button      strong repulsion
//     space             pause / resume the simulation
//     R                 re-seed the cloud
//     + / -             bigger / smaller particles
//     ESC               quit
//
//  If the window is blank or interop fails, see the README: the default is an
//  X11 (XWayland) GL context because GLX is the reliable path for CUDA/OpenGL
//  interop on Linux. CUDA_PARTICLES_PLATFORM=wayland switches to native EGL.
// =============================================================================

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include <vector>

// --- OpenGL, then GLFW, then the CUDA interop header -------------------------
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuda_check.h"
#include "particles.cuh"
#include "shaders.h"

// -----------------------------------------------------------------------------
//  Small GL helpers
// -----------------------------------------------------------------------------
namespace {

GLuint compile_shader(GLenum type, const char* src) {
  GLuint shader = glCreateShader(type);
  glShaderSource(shader, 1, &src, nullptr);
  glCompileShader(shader);

  GLint ok = 0;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetShaderInfoLog(shader, sizeof log, nullptr, log);
    std::fprintf(stderr, "shader compilation failed:\n%s\n", log);
    std::exit(EXIT_FAILURE);
  }
  return shader;
}

GLuint link_program(const char* vertex_src, const char* fragment_src) {
  GLuint program = glCreateProgram();
  GLuint vs = compile_shader(GL_VERTEX_SHADER, vertex_src);
  GLuint fs = compile_shader(GL_FRAGMENT_SHADER, fragment_src);
  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);

  GLint ok = 0;
  glGetProgramiv(program, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetProgramInfoLog(program, sizeof log, nullptr, log);
    std::fprintf(stderr, "program link failed:\n%s\n", log);
    std::exit(EXIT_FAILURE);
  }
  glDeleteShader(vs);
  glDeleteShader(fs);
  return program;
}

void glfw_error_callback(int code, const char* description) {
  std::fprintf(stderr, "[glfw] error %d: %s\n", code, description);
}

// Start GLFW, preferring X11/XWayland because CUDA/GL interop is best supported
// there. Falls back to whatever the platform offers.
bool init_glfw() {
  const char* forced = std::getenv("CUDA_PARTICLES_PLATFORM");
  int platform = GLFW_PLATFORM_X11;
  if (forced && std::strcmp(forced, "wayland") == 0) {
    platform = GLFW_PLATFORM_WAYLAND;
  } else if (forced && std::strcmp(forced, "any") == 0) {
    platform = GLFW_ANY_PLATFORM;
  }

  glfwInitHint(GLFW_PLATFORM, platform);
  if (glfwInit()) return true;

  glfwTerminate();
  glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
  return glfwInit() != 0;
}

// -----------------------------------------------------------------------------
//  Everything the callbacks need lives here, in one place.
// -----------------------------------------------------------------------------
struct App {
  GLFWwindow* window = nullptr;

  double mouse_x = 640.0;   // cursor, in window pixels
  double mouse_y = 360.0;

  int framebuffer_w = 1280;
  int framebuffer_h = 720;

  bool attract = false;   // left button held
  bool repel = false;     // right button held
  bool paused = false;
  bool reset = true;      // re-seed on the very first frame
  bool demo = false;
  bool hud = true;
  bool bloom = true;
  int scene = 0;
  int palette = 0;
  double simulation_time = 0;

  float point_size = 1.6f;  // base point size in pixels
  float fade = 0.035f;      // decay at 60 Hz; converted to elapsed time
  float exposure = 0.8f;

  unsigned seed = 1;

  int frame_count = 0;
  double fps_timer = 0.0;
  double fps = 0.0;
};

App g;

void on_cursor(GLFWwindow*, double x, double y) {
  g.mouse_x = x;
  g.mouse_y = y;
}

void on_mouse_button(GLFWwindow*, int button, int action, int /*mods*/) {
  const bool down = action != GLFW_RELEASE;
  if (button == GLFW_MOUSE_BUTTON_LEFT) g.attract = down;
  if (button == GLFW_MOUSE_BUTTON_RIGHT) g.repel = down;
}

void on_key(GLFWwindow* window, int key, int /*scancode*/, int action,
            int /*mods*/) {
  if (action != GLFW_PRESS && action != GLFW_REPEAT) return;

  switch (key) {
    case GLFW_KEY_ESCAPE:
      glfwSetWindowShouldClose(window, GLFW_TRUE);
      break;
    case GLFW_KEY_SPACE:
      if (action == GLFW_PRESS) g.paused = !g.paused;
      std::printf("%s\n", g.paused ? "paused" : "running");
      break;
    case GLFW_KEY_1:
    case GLFW_KEY_2:
    case GLFW_KEY_3:
      g.scene = key - GLFW_KEY_1;
      g.palette = g.scene;
      g.reset = true;
      g.demo = false;
      break;
    case GLFW_KEY_A:
      if (action == GLFW_PRESS) g.demo = !g.demo;
      break;
    case GLFW_KEY_C:
      if (action == GLFW_PRESS) g.palette = (g.palette + 1) % 3;
      break;
    case GLFW_KEY_B:
      if (action == GLFW_PRESS) g.bloom = !g.bloom;
      break;
    case GLFW_KEY_H:
      if (action == GLFW_PRESS) g.hud = !g.hud;
      break;
    case GLFW_KEY_R:
      g.reset = true;
      ++g.seed;
      std::printf("re-seeding\n");
      break;
    case GLFW_KEY_EQUAL:
    case GLFW_KEY_KP_ADD:
      g.point_size = fminf(g.point_size * 1.15f, 32.0f);
      break;
    case GLFW_KEY_MINUS:
    case GLFW_KEY_KP_SUBTRACT:
      g.point_size = fmaxf(g.point_size / 1.15f, 0.5f);
      break;
    case GLFW_KEY_UP:
      g.fade = fmaxf(g.fade - 0.01f, 0.0f);
      break;
    case GLFW_KEY_DOWN:
      g.fade = fminf(g.fade + 0.01f, 1.0f);
      break;
    default:
      break;
  }
}

void on_framebuffer_size(GLFWwindow*, int width, int height) {
  g.framebuffer_w = width;
  g.framebuffer_h = height;
  glViewport(0, 0, width, height);
}

void print_usage(const char* program) {
  std::printf(
      "usage: %s [options]\n"
      "  --count N   number of particles          (default 400000)\n"
      "  --fade F    trail decay at 60 Hz, 0..1  (default 0.035)\n"
      "  --size P    base point size in pixels   (default 1.6)\n"
      "  --scene S   orbit, aurora, ink          (default orbit)\n"
      "  --demo      automatic three-scene showcase\n"
      "  --staged    CUDA simulation + CPU uploads; no interop required\n"
      "  --exposure F  tone-map exposure, 0.01..10 (default 0.8)\n"
      "  --width W --height H  window size (default 1280x720)\n"
      "  --frames N  exit after N fixed 1/60-second frames\n"
      "  --capture DIR  write PPM frames to DIR (requires --frames)\n"
      "  --no-hud    hide the overlay\n"
      "\n"
      "controls:\n"
      "  mouse       the cloud follows and swirls around the cursor\n"
      "  LMB / RMB   attract / repel\n"
      "  space       pause      R  re-seed\n"
      "  +/-         point size  up/down  trail length\n"
      "  1/2/3       scene      A autopilot      C palette\n"
      "  B           bloom      H overlay\n"
      "  ESC         quit\n",
      program);
}

}  // namespace

#include "render.h"
#include "hud.h"

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** argv) {
  int count = 400000;
  int width = 1280, height = 720, frames = 0;
  bool staged = false;
  const char* capture = nullptr;

  // Reject malformed values rather than silently turning typos into zero.
  auto number = [&](int& i, double low, double high, bool integer = false) {
    const char* option = argv[i];
    if (++i >= argc) {
      std::fprintf(stderr, "missing value for %s\n", option);
      std::exit(EXIT_FAILURE);
    }
    char* end = nullptr;
    const double value = std::strtod(argv[i], &end);
    if (end == argv[i] || *end || !std::isfinite(value) || value < low || value > high ||
        (integer && std::floor(value) != value)) {
      std::fprintf(stderr, "invalid value for %s: %s (range %g..%g)\n", option, argv[i], low, high);
      std::exit(EXIT_FAILURE);
    }
    return value;
  };

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--count") == 0) {
      count = static_cast<int>(number(i, 1, 10000000, true));
    } else if (std::strcmp(argv[i], "--fade") == 0) {
      g.fade = static_cast<float>(number(i, 0, 1));
    } else if (std::strcmp(argv[i], "--size") == 0) {
      g.point_size = static_cast<float>(number(i, 0.25, 32));
    } else if (std::strcmp(argv[i], "--exposure") == 0) {
      g.exposure = static_cast<float>(number(i, 0.01, 10));
    } else if (std::strcmp(argv[i], "--width") == 0) {
      width = static_cast<int>(number(i, 320, 3840, true));
    } else if (std::strcmp(argv[i], "--height") == 0) {
      height = static_cast<int>(number(i, 240, 2160, true));
    } else if (std::strcmp(argv[i], "--frames") == 0) {
      frames = static_cast<int>(number(i, 1, 36000, true));
    } else if (std::strcmp(argv[i], "--staged") == 0) {
      staged = true;
    } else if (std::strcmp(argv[i], "--demo") == 0) {
      g.demo = true;
    } else if (std::strcmp(argv[i], "--no-hud") == 0) {
      g.hud = false;
    } else if (std::strcmp(argv[i], "--capture") == 0 && i + 1 < argc) {
      capture = argv[++i];
    } else if (std::strcmp(argv[i], "--scene") == 0 && i + 1 < argc) {
      const char* scene = argv[++i];
      if (std::strcmp(scene, "orbit") == 0) g.scene = 0;
      else if (std::strcmp(scene, "aurora") == 0) g.scene = 1;
      else if (std::strcmp(scene, "ink") == 0) g.scene = 2;
      else { std::fprintf(stderr, "unknown scene: %s\n", scene); return EXIT_FAILURE; }
      g.palette = g.scene;
    } else if (std::strcmp(argv[i], "--help") == 0 ||
               std::strcmp(argv[i], "-h") == 0) {
      print_usage(argv[0]);
      return EXIT_SUCCESS;
    } else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[i]);
      print_usage(argv[0]);
      return EXIT_FAILURE;
    }
  }
  if (capture && !frames) {
    std::fprintf(stderr, "--capture requires --frames\n");
    return EXIT_FAILURE;
  }
  if (capture) {
    std::error_code error;
    std::filesystem::create_directories(capture, error);
    if (error || !std::filesystem::is_empty(capture, error) || error) {
      std::fprintf(stderr, "capture directory must be accessible and empty: %s\n", capture);
      return EXIT_FAILURE;
    }
  }

  // ---------------------------------------------------------------------------
  //  1. Window and OpenGL context
  // ---------------------------------------------------------------------------
  glfwSetErrorCallback(glfw_error_callback);
  if (!init_glfw()) {
    std::fprintf(stderr, "could not initialise GLFW — is there a display?\n");
    return EXIT_FAILURE;
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_SAMPLES, 0);       // additive blending already smooths
  glfwWindowHint(GLFW_DEPTH_BITS, 0);
  // Hidden capture windows keep the requested dimensions even under tiling WMs.
  glfwWindowHint(GLFW_VISIBLE, capture ? GLFW_FALSE : GLFW_TRUE);

  g.window = glfwCreateWindow(width, height, "CUDA / Particle Atlas", nullptr, nullptr);
  if (!g.window) {
    std::fprintf(stderr, "could not create a window\n");
    glfwTerminate();
    return EXIT_FAILURE;
  }
  glfwMakeContextCurrent(g.window);
  glfwSwapInterval(frames ? 0 : 1);
  g.mouse_x = width * 0.5;
  g.mouse_y = height * 0.5;

  // Callbacks: GLFW gives us the input, we turn it into simulation parameters.
  glfwSetCursorPosCallback(g.window, on_cursor);
  glfwSetMouseButtonCallback(g.window, on_mouse_button);
  glfwSetKeyCallback(g.window, on_key);
  glfwSetFramebufferSizeCallback(g.window, on_framebuffer_size);
  glfwGetFramebufferSize(g.window, &g.framebuffer_w, &g.framebuffer_h);
  glViewport(0, 0, g.framebuffer_w, g.framebuffer_h);

  std::printf("OpenGL      : %s | %s\n", glGetString(GL_RENDERER),
              glGetString(GL_VERSION));

  // ---------------------------------------------------------------------------
  //  2. Which GPU is drawing this window? Use that one for CUDA.
  //     (On a multi-GPU laptop this is the difference between working and
  //     not working at all — see the hint printed on failure.)
  // ---------------------------------------------------------------------------
  unsigned int device_count = 0;
  int devices[8];
  const cudaError_t pairing = staged ? cudaSuccess :
      cudaGLGetDevices(&device_count, devices, 8, cudaGLDeviceListAll);
  if (staged) { device_count = 1; devices[0] = 0; }
  if (pairing != cudaSuccess || device_count == 0) {
    std::fprintf(stderr,
                 "\nCannot pair CUDA with this OpenGL context (%s).\n"
                 "\n"
                 "On a laptop with hybrid graphics the OpenGL context is often\n"
                 "created on the integrated GPU, where CUDA/OpenGL interop is\n"
                 "impossible. Move it to the NVIDIA GPU and try again:\n"
                 "\n"
                  "    prime-run %s\n"
                  "    # or: __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia %s\n"
                  "    # or use --staged for an explicit copy-based fallback\n"
                  "    # If nvidia-smi fails too, check the driver/library versions.\n"
                 "\n",
                 cudaGetErrorString(pairing), argv[0], argv[0]);
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(devices[0]));

  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, devices[0]));
  std::printf("CUDA device : %s (sm_%d%d, %d SMs)\n", prop.name, prop.major,
              prop.minor, prop.multiProcessorCount);

  // ---------------------------------------------------------------------------
  //  3. OpenGL owns the memory. Two VBOs: positions and colours.
  //     The simulation's velocity array, on the other hand, is never drawn, so
  //     it can live in plain CUDA memory.
  // ---------------------------------------------------------------------------
  GLuint vbo_pos = 0, vbo_col = 0, vao_points = 0;
  glGenBuffers(1, &vbo_pos);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(count) * sizeof(float4),
               nullptr, GL_DYNAMIC_DRAW);

  glGenBuffers(1, &vbo_col);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
  glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(count) * sizeof(uchar4),
               nullptr, GL_DYNAMIC_DRAW);

  glGenVertexArrays(1, &vao_points);
  glBindVertexArray(vao_points);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(uchar4),
                        (void*)0);
  glBindVertexArray(0);

  // The fade pass has no attributes at all, but core-profile OpenGL still
  // insists a Vertex Array Object is bound — so it gets an empty one.
  GLuint vao_empty = 0;
  glGenVertexArrays(1, &vao_empty);

  // ---------------------------------------------------------------------------
  //  4. Hand the VBOs to CUDA. From here on, one pointer, two frameworks.
  // ---------------------------------------------------------------------------
  cudaGraphicsResource* res_pos = nullptr;
  cudaGraphicsResource* res_col = nullptr;
  float4* staged_pos = nullptr;
  uchar4* staged_col = nullptr;
  std::vector<float4> host_pos;
  std::vector<uchar4> host_col;
  if (staged) {
    CUDA_CHECK(cudaMalloc(&staged_pos, static_cast<size_t>(count) * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&staged_col, static_cast<size_t>(count) * sizeof(uchar4)));
    host_pos.resize(count); host_col.resize(count);
    std::printf("render path : STAGED (GPU -> CPU -> OpenGL; slower, not zero-copy)\n");
  } else {
    // Positions are read-modify-write; WriteDiscard would invalidate them.
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res_pos, vbo_pos, cudaGraphicsRegisterFlagsNone));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res_col, vbo_col, cudaGraphicsRegisterFlagsNone));
    std::printf("render path : CUDA/OpenGL zero-copy interop\n");
  }

  float4* d_velocity = nullptr;
  CUDA_CHECK(cudaMalloc(&d_velocity, static_cast<size_t>(count) * sizeof(float4)));

  cudaGraphicsResource* resources[2] = {res_pos, res_col};

  // ---------------------------------------------------------------------------
  //  5. Shaders and render state
  // ---------------------------------------------------------------------------
  GLuint point_program = link_program(kPointVertex, kPointFragment);
  GLuint fade_program = link_program(kFadeVertex, kFadeFragment);
  Renderer renderer;
  renderer.init();
  Hud hud;
  hud.init();

  const GLint u_inv_aspect = glGetUniformLocation(point_program, "uInvAspect");
  const GLint u_point_size = glGetUniformLocation(point_program, "uPointSize");
  const GLint u_intensity = glGetUniformLocation(point_program, "uIntensity");
  const GLint u_fade = glGetUniformLocation(fade_program, "uFade");

  glEnable(GL_BLEND);
  glEnable(GL_PROGRAM_POINT_SIZE);   // required for gl_PointSize to matter
  glDisable(GL_DEPTH_TEST);

  glClearColor(0.02f, 0.02f, 0.03f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  std::printf("particles   : %d  (%.1f MB of vertex data)\n", count,
              (count * (sizeof(float4) + sizeof(uchar4))) / 1048576.0);
  std::printf("\ncontrols: mouse to stir | LMB attract | RMB repel | "
              "space pause | R reset | +/- size | up/down trails\n"
              "          1/2/3 scenes | A autopilot | C palette | B bloom | H HUD | ESC quit\n\n");

  // ---------------------------------------------------------------------------
  //  6. The frame loop
  // ---------------------------------------------------------------------------
  double last_time = glfwGetTime();
  g.fps_timer = last_time;
  int rendered_frames = 0;
  float2 previous_mouse = make_float2(0, 0);
  const char* scene_names[] = {"01 / ORBIT", "02 / AURORA", "03 / INK"};

  while (!glfwWindowShouldClose(g.window)) {
    glfwPollEvents();

    // --- timing --------------------------------------------------------------
    const double now = glfwGetTime();
    float dt = static_cast<float>(now - last_time);
    last_time = now;
    if (frames) dt = 1.0f / 60.0f;  // reproducible capture, independent of disk speed
    if (dt > 1.0f / 30.0f) dt = 1.0f / 30.0f;   // never let a hitch explode
    if (dt < 0.0f) dt = 0.0f;

    // --- cursor and world geometry -------------------------------------------
    int win_w = 1, win_h = 1;
    glfwGetWindowSize(g.window, &win_w, &win_h);
    if (g.framebuffer_w <= 0 || g.framebuffer_h <= 0 || win_w <= 0 || win_h <= 0) {
      glfwWaitEventsTimeout(0.05);  // minimized: no divide by zero, no busy loop
      continue;
    }
    const bool resized = renderer.trails.width != g.framebuffer_w ||
                         renderer.trails.height != g.framebuffer_h;
    renderer.resize(g.framebuffer_w, g.framebuffer_h);
    if (!g.paused) g.simulation_time += dt;
    const float time = static_cast<float>(g.simulation_time);
    if (g.demo) {
      const int scene = static_cast<int>(time / 8.0f) % 3;
      if (scene != g.scene) { g.scene = scene; g.palette = scene; g.reset = true; }
    }

    const float aspect =
        static_cast<float>(g.framebuffer_w) / static_cast<float>(g.framebuffer_h);
    const float2 bounds = make_float2(aspect, 1.0f);

    // Window pixels -> world space. y is flipped because GLFW's origin is the
    // top-left corner while OpenGL's is the bottom-left.
    float2 mouse = make_float2(
        ((static_cast<float>(g.mouse_x) / static_cast<float>(win_w)) * 2.0f - 1.0f) * aspect,
        1.0f - (static_cast<float>(g.mouse_y) / static_cast<float>(win_h)) * 2.0f);

    if (g.demo) mouse = make_float2(0.65f * std::cos(time * 0.7f),
                                   0.42f * std::sin(time * 1.1f));
    float2 mouse_velocity = make_float2(0, 0);
    if (dt > 0 && rendered_frames > 0 && !g.reset && !g.paused) {
      mouse_velocity.x = std::clamp((mouse.x - previous_mouse.x) / dt, -4.0f, 4.0f);
      mouse_velocity.y = std::clamp((mouse.y - previous_mouse.y) / dt, -4.0f, 4.0f);
    }
    previous_mouse = mouse;
    const float attraction = g.repel ? -12.0f : (g.attract ? 12.0f : (g.demo ? 1.5f : 0.8f));
    const bool advance = !g.paused || g.reset || resized;

    // --- run the simulation, straight into OpenGL's buffers ------------------
    if (advance) {
      if (!staged) CUDA_CHECK(cudaGraphicsMapResources(2, resources, 0));

      float4* d_pos = staged_pos;
      uchar4* d_col = staged_col;
      size_t mapped_bytes = 0;
      if (!staged) {
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &mapped_bytes, res_pos));
        if (mapped_bytes < static_cast<size_t>(count) * sizeof(float4)) {
          std::fprintf(stderr, "mapped position buffer is too small\n");
          return EXIT_FAILURE;
        }
        CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_col, &mapped_bytes, res_col));
        if (mapped_bytes < static_cast<size_t>(count) * sizeof(uchar4)) {
          std::fprintf(stderr, "mapped colour buffer is too small\n");
          return EXIT_FAILURE;
        }
      }

      if (g.reset) {
        g.reset = false;
        particles_init(d_pos, d_velocity, d_col, count, g.seed, bounds, g.scene, 0);
        renderer.trails.clear();
      }
      if (!g.paused) {
        Params params{};
        params.mouse = mouse;
        params.bounds = bounds;
        params.mouse_velocity = mouse_velocity;
        const int steps = std::max(1, static_cast<int>(std::ceil(dt * 120.0f)));
        params.dt = dt / steps;
        params.time = time;
        params.attraction = attraction;
        params.swirl = 0.85f;
        params.damping = 0.12f;
        params.jitter = 0.15f;
        params.radius = 0.30f;
        params.count = count;
        params.scene = g.scene;
        params.palette = g.palette;

        for (int step = 0; step < steps; ++step) {
          params.time = time - dt + params.dt * (step + 1);
          particles_update(d_pos, d_velocity, d_col, params, 0);
        }
      }

      if (staged) {
        CUDA_CHECK(cudaMemcpy(host_pos.data(), d_pos, host_pos.size() * sizeof(float4), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(host_col.data(), d_col, host_col.size() * sizeof(uchar4), cudaMemcpyDeviceToHost));
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
        glBufferSubData(GL_ARRAY_BUFFER, 0, host_pos.size() * sizeof(float4), host_pos.data());
        glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
        glBufferSubData(GL_ARRAY_BUFFER, 0, host_col.size() * sizeof(uchar4), host_col.data());
      } else {
        CUDA_CHECK(cudaGraphicsUnmapResources(2, resources, 0));
      }
    }

    // --- trails: fade the previous frame slightly toward black ---------------
    // Pausing freezes both the simulation and its light history.
    if (advance) {
      renderer.trails.bind();
      glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
      glUseProgram(fade_program);
      const float frame_fade = 1.0f - std::pow(1.0f - g.fade, dt * 60.0f);
      glUniform1f(u_fade, frame_fade);
      glBindVertexArray(vao_empty);
      glDrawArrays(GL_TRIANGLES, 0, 3);

      // --- the particles: additive, so overlaps bloom --------------------------
      glBlendFunc(GL_SRC_ALPHA, GL_ONE);
      glUseProgram(point_program);
      glUniform1f(u_inv_aspect, 1.0f / aspect);
      glUniform1f(u_point_size,
                  g.point_size * (static_cast<float>(g.framebuffer_h) / 720.0f));
      glUniform1f(u_intensity, std::pow(200000.0f / count, 0.7f) *
                  (g.paused ? 1.0f : dt * 60.0f));
      glBindVertexArray(vao_points);
      glDrawArrays(GL_POINTS, 0, count);
    }
    renderer.draw(vao_empty, g.exposure, g.bloom);
    if (g.hud) hud.draw(g.framebuffer_w, g.framebuffer_h, scene_names[g.scene], count, g.demo, g.paused, staged);
    const GLenum gl_error = glGetError();
    if (gl_error != GL_NO_ERROR) {
      std::fprintf(stderr, "OpenGL error: 0x%x\n", gl_error);
      return EXIT_FAILURE;
    }

    // Read back only for recording; normal interop frames stay GPU-resident.
    // Store every other 60 Hz simulation frame: compact, smooth 30 fps media.
    if (capture && rendered_frames % 2 == 0) {
      const int w = g.framebuffer_w, h = g.framebuffer_h;
      std::vector<unsigned char> pixels(static_cast<size_t>(w) * h * 3);
      glPixelStorei(GL_PACK_ALIGNMENT, 1);
      glReadBuffer(GL_BACK);
      glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());
      if (glGetError() != GL_NO_ERROR) {
        std::fprintf(stderr, "framebuffer readback failed\n");
        return EXIT_FAILURE;
      }
      char name[64];
      std::snprintf(name, sizeof name, "frame-%05d.ppm", rendered_frames / 2);
      const auto path = std::filesystem::path(capture) / name;
      FILE* file = std::fopen(path.c_str(), "wb");
      if (!file) { std::perror("capture"); return EXIT_FAILURE; }
      bool ok = std::fprintf(file, "P6\n%d %d\n255\n", w, h) > 0;
      for (int y = h - 1; y >= 0; --y)
        ok &= std::fwrite(pixels.data() + static_cast<size_t>(y) * w * 3, 1, w * 3, file) == static_cast<size_t>(w * 3);
      if (std::fclose(file) != 0 || !ok) { std::fprintf(stderr, "capture write failed\n"); return EXIT_FAILURE; }
    }

    // --- FPS in the title bar -------------------------------------------------
    g.frame_count++;
    if (now - g.fps_timer >= 0.5) {
      g.fps = g.frame_count / (now - g.fps_timer);
      g.frame_count = 0;
      g.fps_timer = now;

      char title[256];
      std::snprintf(title, sizeof title,
                    "CUDA / Particle Atlas — %s — %d particles — %.0f FPS%s",
                    scene_names[g.scene], count, g.fps, g.paused ? " — paused" : "");
      glfwSetWindowTitle(g.window, title);
    }

    glfwSwapBuffers(g.window);
    ++rendered_frames;
    if (frames && rendered_frames >= frames) break;
  }

  // ---------------------------------------------------------------------------
  //  7. Clean up (unregister before deleting the GL buffers!)
  // ---------------------------------------------------------------------------
  if (!staged) {
    CUDA_CHECK(cudaGraphicsUnregisterResource(res_pos));
    CUDA_CHECK(cudaGraphicsUnregisterResource(res_col));
  } else {
    CUDA_CHECK(cudaFree(staged_pos));
    CUDA_CHECK(cudaFree(staged_col));
  }
  CUDA_CHECK(cudaFree(d_velocity));

  glDeleteProgram(point_program);
  glDeleteProgram(fade_program);
  renderer.destroy();
  hud.destroy();
  glDeleteBuffers(1, &vbo_pos);
  glDeleteBuffers(1, &vbo_col);
  glDeleteVertexArrays(1, &vao_points);
  glDeleteVertexArrays(1, &vao_empty);

  glfwDestroyWindow(g.window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
