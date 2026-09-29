// -----------------------------------------------------------------------------
//  06 — CUDA <-> OpenGL interop (the bridge to the final project)
// -----------------------------------------------------------------------------
//  Build & run:   make run-06_interop
//
//  The particle system at the end of this repo works because CUDA and OpenGL
//  can share the very same buffer. You do not copy anything: CUDA writes
//  vertices straight into OpenGL's vertex buffer object (VBO), and OpenGL
//  draws from it on the next frame. Zero round-trips through the CPU.
//
//  The recipe, in five lines, repeated every frame:
//
//      cudaGraphicsMapResources(...)            // give CUDA access
//      cudaGraphicsResourceGetMappedPointer(..) // hand me the device pointer
//      kernel<<<...>>>(that pointer)            // write the data
//      cudaGraphicsUnmapResources(...)          // hand it back to OpenGL
//      glDrawArrays(...)                        // draw
//
//  This example animates a single triangle. The final project is exactly this
//  with 400,000 vertices instead of 3 — the structure does not change at all.
// -----------------------------------------------------------------------------

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// OpenGL headers first, then the CUDA interop header.
#define GL_GLEXT_PROTOTYPES
#include <GL/gl.h>
#include <GL/glext.h>
#include <GLFW/glfw3.h>

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include "cuda_check.h"

// -----------------------------------------------------------------------------
//  GLSL — three vertices in, a triangle out. Same shaders the particles use.
// -----------------------------------------------------------------------------
static const char* kVertexShader = R"GLSL(
#version 330 core
layout(location = 0) in vec4 aPos;    // xy = position, z/w unused here
layout(location = 1) in vec4 aCol;    // rgba, 0..1
out vec4 vCol;
void main() {
    gl_Position = vec4(aPos.xy, 0.0, 1.0);
    vCol = aCol;
}
)GLSL";

static const char* kFragmentShader = R"GLSL(
#version 330 core
in vec4 vCol;
out vec4 fragColor;
void main() { fragColor = vCol; }
)GLSL";

static GLuint compile(GLenum type, const char* src) {
  GLuint s = glCreateShader(type);
  glShaderSource(s, 1, &src, nullptr);
  glCompileShader(s);
  GLint ok = 0;
  glGetShaderiv(s, GL_COMPILE_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetShaderInfoLog(s, sizeof log, nullptr, log);
    std::fprintf(stderr, "shader compile failed:\n%s\n", log);
    std::exit(EXIT_FAILURE);
  }
  return s;
}

static GLuint make_program(const char* vs, const char* fs) {
  GLuint p = glCreateProgram();
  GLuint v = compile(GL_VERTEX_SHADER, vs);
  GLuint f = compile(GL_FRAGMENT_SHADER, fs);
  glAttachShader(p, v);
  glAttachShader(p, f);
  glLinkProgram(p);
  GLint ok = 0;
  glGetProgramiv(p, GL_LINK_STATUS, &ok);
  if (!ok) {
    char log[4096];
    glGetProgramInfoLog(p, sizeof log, nullptr, log);
    std::fprintf(stderr, "program link failed:\n%s\n", log);
    std::exit(EXIT_FAILURE);
  }
  glDeleteShader(v);
  glDeleteShader(f);
  return p;
}

// -----------------------------------------------------------------------------
//  The kernel: three threads, one triangle, animated by time.
// -----------------------------------------------------------------------------
__global__ void animate_triangle(float4* __restrict__ pos,
                                 uchar4* __restrict__ col, float t) {
  const int i = threadIdx.x;
  if (i >= 3) return;

  const float angle = t * 0.9f + i * 2.0943951f;  // 2*pi/3 between corners
  pos[i] = make_float4(__cosf(angle) * 0.6f, __sinf(angle) * 0.6f, 0.f, 1.f);

  col[i] = (i == 0) ? make_uchar4(255, 90, 130, 255)
          : (i == 1) ? make_uchar4(110, 230, 255, 255)
                     : make_uchar4(255, 225, 110, 255);
}

int main() {
  // ---------------------------------------------------------------------------
  //  1. A window. We ask GLFW for X11/XWayland: GLX is the well-trodden path
  //     for CUDA/GL interop on Linux. Set CUDA_PARTICLES_PLATFORM=wayland to
  //     try a native Wayland (EGL) context instead.
  // ---------------------------------------------------------------------------
  const char* forced = std::getenv("CUDA_PARTICLES_PLATFORM");
  const bool want_wayland = forced && std::strcmp(forced, "wayland") == 0;
  glfwInitHint(GLFW_PLATFORM,
               want_wayland ? GLFW_PLATFORM_WAYLAND : GLFW_PLATFORM_X11);
  if (!glfwInit()) {
    glfwTerminate();
    glfwInitHint(GLFW_PLATFORM, GLFW_ANY_PLATFORM);
    if (!glfwInit()) {
      std::fprintf(stderr, "glfwInit failed: no display?\n");
      return EXIT_FAILURE;
    }
  }

  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

  GLFWwindow* window = glfwCreateWindow(960, 720, "06 — CUDA / OpenGL interop",
                                        nullptr, nullptr);
  if (!window) {
    std::fprintf(stderr, "could not create a window\n");
    glfwTerminate();
    return EXIT_FAILURE;
  }
  glfwMakeContextCurrent(window);
  glfwSwapInterval(1);

  std::printf("GL vendor   : %s\n", glGetString(GL_VENDOR));
  std::printf("GL renderer : %s\n", glGetString(GL_RENDERER));
  std::printf("GL version  : %s\n", glGetString(GL_VERSION));

  // ---------------------------------------------------------------------------
  //  2. Ask CUDA which GPU is driving this OpenGL context.
  // ---------------------------------------------------------------------------
  unsigned int device_count = 0;
  int devices[8];
  CUDA_CHECK(cudaGLGetDevices(&device_count, devices, 8, cudaGLDeviceListAll));
  if (device_count == 0) {
    std::fprintf(stderr, "no CUDA device is associated with this GL context\n");
    return EXIT_FAILURE;
  }
  CUDA_CHECK(cudaSetDevice(devices[0]));

  // ---------------------------------------------------------------------------
  //  3. Create the GL buffers, then register them with CUDA.
  // ---------------------------------------------------------------------------
  const int count = 3;

  GLuint vbo_pos = 0, vbo_col = 0, vao = 0;
  glGenBuffers(1, &vbo_pos);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
  glBufferData(GL_ARRAY_BUFFER, count * sizeof(float4), nullptr, GL_DYNAMIC_DRAW);

  glGenBuffers(1, &vbo_col);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
  glBufferData(GL_ARRAY_BUFFER, count * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);

  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_pos);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, sizeof(float4), (void*)0);
  glBindBuffer(GL_ARRAY_BUFFER, vbo_col);
  glEnableVertexAttribArray(1);
  glVertexAttribPointer(1, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(uchar4), (void*)0);
  glBindVertexArray(0);

  cudaGraphicsResource *res_pos = nullptr, *res_col = nullptr;
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res_pos, vbo_pos,
                                          cudaGraphicsMapFlagsNone));
  CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&res_col, vbo_col,
                                          cudaGraphicsMapFlagsNone));

  GLuint program = make_program(kVertexShader, kFragmentShader);

  // ---------------------------------------------------------------------------
  //  4. Frame loop.
  // ---------------------------------------------------------------------------
  const double start_time = glfwGetTime();
  cudaGraphicsResource* resources[2] = {res_pos, res_col};

  while (!glfwWindowShouldClose(window)) {
    glfwPollEvents();
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
      glfwSetWindowShouldClose(window, GLFW_TRUE);
    }
    const float t = static_cast<float>(glfwGetTime() - start_time);

    // --- map both buffers into CUDA -----------------------------------------
    CUDA_CHECK(cudaGraphicsMapResources(2, resources, 0));

    float4* d_pos = nullptr;
    uchar4* d_col = nullptr;
    size_t bytes = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_pos, &bytes, res_pos));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_col, &bytes, res_col));

    // --- compute straight into OpenGL's memory ------------------------------
    animate_triangle<<<1, 4>>>(d_pos, d_col, t);
    CUDA_CHECK(cudaGetLastError());

    // --- hand the buffers back ----------------------------------------------
    CUDA_CHECK(cudaGraphicsUnmapResources(2, resources, 0));

    // --- draw ---------------------------------------------------------------
    int fb_w = 0, fb_h = 0;
    glfwGetFramebufferSize(window, &fb_w, &fb_h);
    glViewport(0, 0, fb_w, fb_h);

    glClearColor(0.03f, 0.03f, 0.05f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(program);
    glBindVertexArray(vao);
    glDrawArrays(GL_TRIANGLES, 0, count);

    glfwSwapBuffers(window);
  }

  // ---------------------------------------------------------------------------
  //  5. Clean up: unregister before deleting the GL buffers.
  // ---------------------------------------------------------------------------
  CUDA_CHECK(cudaGraphicsUnregisterResource(res_pos));
  CUDA_CHECK(cudaGraphicsUnregisterResource(res_col));
  glDeleteBuffers(1, &vbo_pos);
  glDeleteBuffers(1, &vbo_col);
  glDeleteVertexArrays(1, &vao);
  glDeleteProgram(program);
  glfwDestroyWindow(window);
  glfwTerminate();
  return EXIT_SUCCESS;
}
