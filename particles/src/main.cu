// =============================================================================
//  CUDA Particles — the final project
// =============================================================================
//  Build & run:   make run-particles
//                 make run-particles ARGS="--count 1000000 --fade 0.05"
//
//  A GPU-resident particle system you stir with the mouse. The simulation
//  lives entirely in CUDA; the pixels go straight to the screen through
//  OpenGL. Between them there is not a single byte copied by the CPU — CUDA
//  writes into OpenGL's own vertex buffers via the interop you met in
//  example 06.
//
//  Everything you learned on the way here is used:
//
//    01 hello              kernel launches, thread/block indices
//    02 vector_add         cudaMalloc / memcpy, error checking
//    03 index_math         one thread per element, launch geometry
//    05 timing_streams     frame timing (watch the FPS in the title bar)
//    06 interop            cudaGraphicsMap / Unmap, shared VBOs
//
//  What is new here is only the loop that ties them together.
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

  double mouse_x = 0.0;   // cursor, in window pixels
  double mouse_y = 0.0;

  int framebuffer_w = 1280;
  int framebuffer_h = 720;

  bool attract = false;   // left button held
  bool repel = false;     // right button held
  bool paused = false;
  bool reset = true;      // re-seed on the very first frame

  float point_size = 2.0f;  // base point size in pixels
  float fade = 0.09f;       // trail length: bigger = shorter trails

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
      g.paused = !g.paused;
      std::printf("%s\n", g.paused ? "paused" : "running");
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
      "usage: %s [--count N] [--fade F] [--size P]\n"
      "  --count N   number of particles          (default 400000)\n"
      "  --fade F    trail fade per frame, 0..1   (default 0.09)\n"
      "  --size P    base point size in pixels    (default 2.0)\n"
      "\n"
      "controls:\n"
      "  mouse       the cloud follows and swirls around the cursor\n"
      "  LMB / RMB   attract / repel\n"
      "  space       pause      R  re-seed\n"
      "  +/-         point size  up/down  trail length\n"
      "  ESC         quit\n",
      program);
}

}  // namespace

// =============================================================================
//  main
// =============================================================================
int main(int argc, char** argv) {
  int count = 400000;

  for (int i = 1; i < argc; ++i) {
    if (std::strcmp(argv[i], "--count") == 0 && i + 1 < argc) {
      count = std::atoi(argv[++i]);
    } else if (std::strcmp(argv[i], "--fade") == 0 && i + 1 < argc) {
      g.fade = static_cast<float>(std::atof(argv[++i]));
    } else if (std::strcmp(argv[i], "--size") == 0 && i + 1 < argc) {
      g.point_size = static_cast<float>(std::atof(argv[++i]));
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
  if (count < 1) count = 1;
  g.fade = fminf(fmaxf(g.fade, 0.0f), 1.0f);
  g.point_size = fmaxf(g.point_size, 0.25f);

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

  g.window = glfwCreateWindow(1280, 720, "CUDA Particles", nullptr, nullptr);
  if (!g.window) {
    std::fprintf(stderr, "could not create a window\n");
    glfwTerminate();
    return EXIT_FAILURE;
  }
  glfwMakeContextCurrent(g.window);
  glfwSwapInterval(1);

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
  //     not working at all.)
  // ---------------------------------------------------------------------------
  unsigned int device_count = 0;
  int devices[8];
  CUDA_CHECK(cudaGLGetDevices(&device_count, devices, 8, cudaGLDeviceListAll));
  if (device_count == 0) {
    std::fprintf(stderr, "no CUDA device is associated with this GL context\n");
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
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res_pos, vbo_pos,
                                          cudaGraphicsMapFlagsNone));
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res_col, vbo_col,
                                          cudaGraphicsMapFlagsNone));

  float4* d_velocity = nullptr;
  CUDA_CHECK(cudaMalloc(&d_velocity, static_cast<size_t>(count) * sizeof(float4)));

  cudaGraphicsResource* resources[2] = {res_pos, res_col};

  // ---------------------------------------------------------------------------
  //  5. Shaders and render state
  // ---------------------------------------------------------------------------
  GLuint point_program = link_program(kPointVertex, kPointFragment);
  GLuint fade_program = link_program(kFadeVertex, kFadeFragment);

  const GLint u_inv_aspect = glGetUniformLocation(point_program, "uInvAspect");
  const GLint u_point_size = glGetUniformLocation(point_program, "uPointSize");
  const GLint u_fade = glGetUniformLocation(fade_program, "uFade");

  glEnable(GL_BLEND);
  glEnable(GL_PROGRAM_POINT_SIZE);   // required for gl_PointSize to matter
  glDisable(GL_DEPTH_TEST);

  glClearColor(0.02f, 0.02f, 0.03f, 1.0f);
  glClear(GL_COLOR_BUFFER_BIT);

  std::printf("particles   : %d  (%.1f MB of vertex data)\n", count,
              (count * (sizeof(float4) + sizeof(uchar4))) / 1048576.0);
  std::printf("\ncontrols: mouse to stir | LMB attract | RMB repel | "
              "space pause | R reset | +/- size | up/down trails | ESC quit\n\n");

  // ---------------------------------------------------------------------------
  //  6. The frame loop
  // ---------------------------------------------------------------------------
  double last_time = glfwGetTime();
  g.fps_timer = last_time;

  while (!glfwWindowShouldClose(g.window)) {
    glfwPollEvents();

    // --- timing --------------------------------------------------------------
    const double now = glfwGetTime();
    float dt = static_cast<float>(now - last_time);
    last_time = now;
    if (dt > 1.0f / 30.0f) dt = 1.0f / 30.0f;   // never let a hitch explode
    if (dt < 0.0f) dt = 0.0f;

    // --- cursor and world geometry -------------------------------------------
    int win_w = 1, win_h = 1;
    glfwGetWindowSize(g.window, &win_w, &win_h);

    const float aspect =
        static_cast<float>(g.framebuffer_w) / static_cast<float>(g.framebuffer_h);
    const float2 bounds = make_float2(aspect, 1.0f);

    // Window pixels -> world space. y is flipped because GLFW's origin is the
    // top-left corner while OpenGL's is the bottom-left.
    const float2 mouse = make_float2(
        ((static_cast<float>(g.mouse_x) / static_cast<float>(win_w)) * 2.0f - 1.0f) * aspect,
        1.0f - (static_cast<float>(g.mouse_y) / static_cast<float>(win_h)) * 2.0f);

    const float attraction = g.repel ? -18.0f : (g.attract ? 18.0f : 3.5f);

    // --- run the simulation, straight into OpenGL's buffers ------------------
    if (!g.paused) {
      CUDA_CHECK(cudaGraphicsMapResources(2, resources, 0));

      float4* d_pos = nullptr;
      uchar4* d_col = nullptr;
      size_t mapped_bytes = 0;
      CUDA_CHECK(
          cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &mapped_bytes, res_pos));
      CUDA_CHECK(
          cudaGraphicsResourceGetMappedPointer((void**)&d_col, &mapped_bytes, res_col));

      if (g.reset) {
        g.reset = false;
        particles_init(d_pos, d_velocity, d_col, count, g.seed, bounds, 0);
      } else {
        Params params{};
        params.mouse = mouse;
        params.bounds = bounds;
        params.dt = dt;
        params.time = static_cast<float>(now);
        params.attraction = attraction;
        params.swirl = 1.9f;
        params.damping = 1.15f;
        params.jitter = 0.7f;
        params.radius = 0.30f;
        params.count = count;

        particles_update(d_pos, d_velocity, d_col, params, 0);
      }

      CUDA_CHECK(cudaGraphicsUnmapResources(2, resources, 0));
    }

    // --- trails: fade the previous frame slightly toward black ---------------
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glUseProgram(fade_program);
    glUniform1f(u_fade, g.fade);
    glBindVertexArray(vao_empty);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    // --- the particles: additive, so overlaps bloom --------------------------
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glUseProgram(point_program);
    glUniform1f(u_inv_aspect, 1.0f / aspect);
    glUniform1f(u_point_size,
                g.point_size * (static_cast<float>(g.framebuffer_h) / 720.0f));
    glBindVertexArray(vao_points);
    glDrawArrays(GL_POINTS, 0, count);

    // --- FPS in the title bar -------------------------------------------------
    g.frame_count++;
    if (now - g.fps_timer >= 0.5) {
      g.fps = g.frame_count / (now - g.fps_timer);
      g.frame_count = 0;
      g.fps_timer = now;

      char title[256];
      std::snprintf(title, sizeof title,
                    "CUDA Particles — %d particles — %.0f FPS%s",
                    count, g.fps, g.paused ? " — paused" : "");
      glfwSetWindowTitle(g.window, title);
    }

    glfwSwapBuffers(g.window);
  }

  // ---------------------------------------------------------------------------
  //  7. Clean up (unregister before deleting the GL buffers!)
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaGraphicsUnregisterResource(res_pos));
  CUDA_CHECK(cudaGraphicsUnregisterResource(res_col));
  CUDA_CHECK(cudaFree(d_velocity));

  glDeleteProgram(point_program);
  glDeleteProgram(fade_program);
  glDeleteBuffers(1, &vbo_pos);
  glDeleteBuffers(1, &vbo_col);
  glDeleteVertexArrays(1, &vao_points);
  glDeleteVertexArrays(1, &vao_empty);

  glfwDestroyWindow(g.window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
